import os
import numpy as np
from scipy.spatial import KDTree
import glob


def create_tree(array):
    """
    根据点集建立kd_tree
    """
    tree = KDTree(array)
    return tree


def get_gum_line_pts(gum_line_path):
    """
    读取牙龈线文件，pts格式
    """
    f = open(gum_line_path)
    pts = []

    is_generate_by_streamflow = False  # 是否由前台界面生成

    for num, line in enumerate(f):
        if 0 == num and line.strip() == "BEGIN_0":
            is_generate_by_streamflow = True
            continue
        if line.strip() == "BEGIN" or line.strip() == "END":
            continue
        if is_generate_by_streamflow:
            line = line.strip()
            if line == "END_0":
                pts.append(pts[0])
            else:
                splitted_line = line.split()
                point = [float(i) for i in splitted_line][:3]  # 只取点坐标，去掉法向量
                assert len(point) == 3, "点的坐标为x,y,z"
                pts.append(point)
        else:
            line = line.strip()
            splitted_line = line.split()
            point = [float(i) for i in splitted_line]
            assert len(point) == 3, "点的坐标为x,y,z"
            pts.append(point)

    f.close()
    pts = np.asarray(pts)
    return pts


def calculate_dist(tree, predict_pts):
    max_dist = 0
    max_index = 0
    for idx, predict_pt in enumerate(predict_pts):
        dist, index = tree.query(predict_pt, k=1)
        if dist > max_dist:
            max_dist = dist
            max_index = idx

    return max_dist, max_index


def get_max_dist(target_pts_file, predict_pts_file):
    target_pts = get_gum_line_pts(target_pts_file)
    predict_pts = get_gum_line_pts(predict_pts_file)

    tree = create_tree(target_pts)

    if len(predict_pts) == 0:
        return np.inf, np.inf   # 无结果

    max_dist, max_index = calculate_dist(tree, predict_pts)

    return max_dist, predict_pts[max_index]


def show_predict(model_file, pred_pts, target_pts, max_dist_pts=None):
    from vedo import load, show, Point
    import vtkplotter as vtkp

    a = load(model_file).c(('magenta'))
    if os.path.splitext(pred_pts)[-1] == ".vtk":
        p = load(pred_pts).pointSize(10).c(('green'))
    else:
        predict_pts = get_gum_line_pts(pred_pts)
        p = vtkp.Points(predict_pts.reshape(-1, 3))
        p = p.pointSize(10).c(('green'))

    if os.path.splitext(target_pts)[-1] == ".vtk":
        t = load(target_pts).pointSize(5).c(("red"))
    else:
        target_pts = get_gum_line_pts(target_pts)
        t = vtkp.Points(target_pts.reshape(-1, 3))
        t = t.pointSize(5).c(("red"))

    if max_dist_pts:
        p1 = Point(max_dist_pts, r=15, c='yellow')
        show(a, p, t, p1)
    else:
        show(a, p, t)


if __name__ == "__main__":
    target_pts_dir = "/run/user/1000/gvfs/smb-share:server=10.99.11.210,share=meshcnn/Test_5044/pts"   # 正确牙龈线路径
    predict_model_dir = "/home/heygears/work/test_eval/predict_obj"                                    # 预测模型路径
    predict_pts_dir = "/home/heygears/work/test_eval/predict_pts"                                      # 预测牙龈线路径
    threshold = 2                                                                                      # 距离阈值

    predict_pts_files = glob.glob(os.path.join(predict_pts_dir, "*.pts"))
    error_models = {}
    for idx, pred_pts_file in enumerate(predict_pts_files):
        base_name = os.path.basename(pred_pts_file)
        target_pts_file = os.path.join(target_pts_dir, base_name)
        if os.path.isfile(target_pts_file):
            max_dist, max_pts = get_max_dist(target_pts_file, pred_pts_file)
            if max_dist is np.inf and max_pts is np.inf:
                continue
            if max_dist > threshold:
                error_models[os.path.splitext(os.path.basename(base_name))[0]] = [max_dist, max_pts.tolist()]
            print("{}, file: {}, dist: {}".format(idx + 1, base_name, max_dist))

    print("******一共{}个模型的距离超过阈值{}，占总模型数{}的百分比为：{}*********\n".
          format(len(error_models), threshold, len(predict_pts_files), len(error_models) / len(predict_pts_files)))

    # show model
    for model_name in error_models.keys():
        model_file = os.path.join(predict_model_dir, model_name + ".obj")
        target_pts_file = os.path.join(target_pts_dir, model_name + ".pts")
        predict_pts_file = os.path.join(predict_pts_dir, model_name + ".pts")

        if os.path.isfile(model_file) and os.path.isfile(target_pts_file) and os.path.isfile(predict_pts_file):
            print("file path is: {}, dist: {}".format(model_file, error_models[model_name][0]))
            show_predict(model_file, predict_pts_file, target_pts_file, error_models[model_name][1])

