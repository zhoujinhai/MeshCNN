import glob
import os
from vedo import load, show
import vtkplotter as vtkp
import numpy as np
import random


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


def show_pts(stl_list, pts_path):
    # show pts data
    for i, stl in enumerate(stl_list):
        file_name = os.path.basename(stl)[:-4]
        pts = os.path.join(pts_path, file_name + ".pts")
        gum_line = get_gum_line_pts(pts)
        print(i + 1, " ", file_name)
        a = load(stl).c(('magenta'))
        p = vtkp.Points(gum_line.reshape(-1, 3))
        p = p.pointSize(10).c(('green'))
        show(a, p)


def show_model(dir_path, mode=1, file_name=None):
    pts_path = os.path.join(dir_path, "pts")
    stl_path = os.path.join(dir_path, "stl")

    stl_list = glob.glob(os.path.join(stl_path, "*.stl"))

    if mode == 1:    # 显示所有数据
        show_pts(stl_list, pts_path)
    elif mode == 2:
        random_stl_list = random.sample(stl_list, 10)
        show_pts(random_stl_list, pts_path)
    elif mode == 3:
        show_stl_list = []
        for file_basename in file_name:
            show_stl_list.append(os.path.join(stl_path, file_basename + ".stl"))
        show_pts(show_stl_list, pts_path)
    else:
        print("请选择正确的模式，1: 显示所有，2: 随机显示10个，3: 显示指定文件")


if __name__ == "__main__":

    dir_path = "/run/user/1000/gvfs/smb-share:server=10.99.11.210,share=meshcnn/Test_5044/"  # pts和stl文件存放目录
    file_name = ["1_459650_1_U_15", "8YZRK_VS_SET_VSc3_Subsetup10_Mandibular"]   # 指定显示的文件名
    # file_name = []
    # with open("./error_label.txt", "r") as f:
    #     for line in f:
    #         line = line.strip()
    #         file_name.append(line)
    mode = 3  # 1: 查看全部， 2: 随机查看10个， 3: 查看指定文件
    show_model(dir_path, mode, file_name)
