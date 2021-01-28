import glob
import os
from vedo import load, show
import numpy as np


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


def pts_to_vtk(gum_line_pts, save_path="./test.vtk"):
    """
    将牙龈点pts格式转为vtk格式
    @pts: 点集 [[x, y, z], [x, y, z], ...]
    @save_path: 保存路径
    return: None
    """
    import vtkplotter as vtkp
    vtk_point = vtkp.Points(gum_line_pts.reshape(-1, 3))
    vtkp.write(vtk_point, save_path, binary=False)
    print("vtk file is saved in ", save_path)


if __name__ == "__main__":
    # # # convert pts to vtk in train Data
    pts_path = "/run/user/1000/gvfs/smb-share:server=10.99.11.210,share=meshcnn/Pts_5044"
    stl_path = "/run/user/1000/gvfs/smb-share:server=10.99.11.210,share=meshcnn/Pts_5044"
    # obj_path = "/home/heygears/work/Tooth_data_prepare/tooth/three_batch_data/file/train"
    vtk_path = "/run/user/1000/gvfs/smb-share:server=10.99.11.210,share=meshcnn/Pts_5044_vtk"
    pts_save_path = "/run/user/1000/gvfs/smb-share:server=10.99.11.210,share=meshcnn/Pts_5044_pts"
    pts_list = glob.glob(os.path.join(pts_path, "*.pts"))
    stl_list = glob.glob(os.path.join(stl_path, "*.stl"))
    # obj_list = glob.glob(os.path.join(obj_path, "*.obj"))
    if not os.path.isdir(vtk_path):
        os.makedirs(vtk_path)
    if not os.path.isdir(pts_save_path):
        os.makedirs(pts_save_path)

    for pts in pts_list:
        file_name = os.path.basename(pts)[:-4]
        save_vtk_path = os.path.join(vtk_path, file_name+".vtk")
        save_pts_path = os.path.join(pts_save_path, file_name + ".pts")
        print("*****", file_name)
        gum_line = get_gum_line_pts(pts)
        np.savetxt(save_pts_path, gum_line)
        pts_to_vtk(gum_line, save_vtk_path)

    # show pts data
    vtk_list = glob.glob(os.path.join(vtk_path, "*.vtk"))
    for i, stl in enumerate(stl_list):
    # for i, stl in enumerate(obj_list):
            file_name = os.path.basename(stl)[:-4]
            vtk = os.path.join(vtk_path, file_name+".vtk")
            print(i+1, " ", file_name)
            a = load(stl).c(('magenta'))
            b = load(vtk).pointSize(10).c(('green'))
            show(a, b)
