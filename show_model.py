import vedo
from vedo import load, show
import os
import glob
import numpy as np
import shutil

# import sys
# sys.path.append(r"/home/heygears/work/package/hgapi")
# import hgapi
from hgapi import hgapi


def get_edges(faces):
    """
    根据面得到相应的边
    @faces: 模型的所有面
    return: 模型的边
    """
    edge2key = dict()
    edges = []
    edges_count = 0
    for face_id, face in enumerate(faces):
        faces_edges = []
        for i in range(3):
            cur_edge = (face[i], face[(i + 1) % 3])
            faces_edges.append(cur_edge)
        for idx, edge in enumerate(faces_edges):
            edge = tuple(sorted(list(edge)))
            if edge not in edge2key:
                edge2key[edge] = edges_count
                edges_count += 1
                edges.append(list(edge))
    return edges


def parse_obje(obj_file):
    """
    解析obj文件， 获取点，边，面
    @obj_file: obj模型文件路径
    return: 模型的点，边，面信息
    """

    vs = []
    faces = []
    edges = []

    with open(obj_file) as f:
        for line in f:
            line = line.strip()
            splitted_line = line.split()
            if not splitted_line:
                continue
            elif splitted_line[0] == 'v':
                vs.append([float(v) for v in splitted_line[1:]])
            elif splitted_line[0] == 'f':
                try:
                    faces.append([int(c) - 1 for c in splitted_line[1:]])
                except ValueError:
                    faces.append([int(c.split('/')[0]) for c in splitted_line[1:]])
            elif splitted_line[0] == 'e':
                if len(splitted_line) >= 4:
                    edge_v = [int(c) - 1 for c in splitted_line[1:-1]]
                    edge_c = int(splitted_line[-1])
                    edge_v.append(edge_c)  # class
                    edges.append(edge_v)
            else:
                continue

    vs = np.array(vs)
    faces = np.array(faces, dtype=int)
    if len(edges) == 0:
        edges = get_edges(faces)
    edges = np.array(edges)

    return vs, faces, edges


def show_predict(predict1, predict2, pts):
    """
    显示预测结果
    @predict1: 牙齿部分
    @predict2: 牙龈部分
    @pts: 牙龈线点
    return: None
    """
    a = load(predict1).c(('blue'))
    b = load(predict2).c(('magenta'))
    c = load(pts).pointSize(10).c(('green'))

    show(a, b, c)


def show_origin(origin1, origin2, pts1, pts2):
    """
    显示原始模型分割结果
    @origin1: 牙齿部分
    @origin2: 牙龈部分
    @pts1: 通过预测点投影所得牙龈线点
    @pts2: 通过边投影所得牙龈线点
    return: None
    """
    a = load(origin1).c(('blue'))
    b = load(origin2).c(('magenta'))
    c = load(pts2).pointSize(10).c(('green'))  # 边投影
    d = load(pts1).pointSize(10).c(('red'))    # 点投影

    show(a, b, c, d)


def stlHandlerRead(fName):
    # 一、找出最大面后旋转,找出正面方向
    mesh = hgapi.Mesh()
    stlHandler = hgapi.STLFileHandler()
    print("stlHandlerRead > Read:", stlHandler.Read(fName, mesh))
    norm1 = hgapi.Vector3()  # 坐标值v1.x(), v1.y(), v1.z()
    centroid1 = hgapi.Vector3()
    od = hgapi.OrientationDetection()
    planarFaceSet = hgapi.IntVector()
    od.FindPlanarBase2(mesh, norm1, centroid1, 0.9995, planarFaceSet)
    rotation1 = hgapi.Matrix4.Rotation(norm1, hgapi.Vector3(0.0, 0.0, -1.0))
    mesh.Transform(rotation1)

    dir = hgapi.Vector2()
    center = hgapi.Vector2()
    od = hgapi.OrientationDetection()
    od.Find2DHeadDirection(mesh, dir, center, True)
    handler = hgapi.STLFileHandler()
    rotation2 = hgapi.Matrix4.Rotation(hgapi.Vector3(dir), hgapi.Vector3(0.0, 1.0, 0.0))
    mesh.Transform(rotation2)

    # 二、平移至0.0 0.0 0.0
    handler = hgapi.STLFileHandler()
    min_bound = hgapi.Vector3(mesh.GetBound().GetMinBound().x(),
                              mesh.GetBound().GetMinBound().y(),
                              mesh.GetBound().GetMinBound().z())
    translation = hgapi.Matrix4.Translation(-min_bound)
    mesh.Transform(translation)
    return mesh


def generatePreview():
    fName = r"/home/heygears/work/Tooth_data_prepare/8AP1S/VS_SET_VSc2_Subsetup8_Maxillar.stl"
    mesh = stlHandlerRead(fName)
    rasterizer = hgapi.Rasterizer()
    print("rasterizer.Init", rasterizer.Init())
    entityArray = hgapi.EntityVector()
    entityArray.push_back(mesh)
    sharedImage = hgapi.SharedImageVector()
    # LEFT(1)，RIGHT(2)，FRONT(4)，BACK(8)，BOTTOM(16)，TOP(32)，TOP_LEFT(64)，TOP_RIGHT(128)，TOP_FRONT(256)，
    # TOP_BACK(512)，BOTTOM_LEFT(1024)，BOTTOM_RIGHT(2048)，BOTTOM_FRONT(4096)，BOTTOM_BACK(8192)
    rasterizer.GeneratePreview(entityArray, 0.1, 32, sharedImage)
    pngHandler = hgapi.PNGHandler()
    print("Export:", pngHandler.Export("test.png", sharedImage[0]))


def simplify():
    file_name = "/home/heygears/work/Tooth_data_prepare/tooth/stl/1PT7E_VS_SET_VSc2_Subsetup11_Mandibular.stl"
    mesh = hgapi.Mesh()
    stlHandler = hgapi.STLFileHandler()
    print("stlHandlerRead > Read:", stlHandler.Read(file_name, mesh))
    hgapi.MeshSimplify_Simplify(mesh, 5000)
    print("simplify: ", stlHandler.Write("./down.stl", mesh))


def healing(mesh_path, save_path):
    mesh_healing = hgapi.MeshHealing()
    mesh = hgapi.Mesh()
    stl_handler = hgapi.STLFileHandler()
    stl_handler.Read(mesh_path, mesh)
    mesh_healing.RemoveSmallerIslands(mesh)
    mesh_healing.RemoveDanglingFace(mesh)
    stl_handler.Write(save_path, mesh)


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
    # a = load('/home/heygears/work/Tooth_data_prepare/tooth/8AP1S/mesh1.obj').c(('blue'))
    # b = load('/home/heygears/work/Tooth_data_prepare/tooth/8AP1S/mesh2.obj').c(('magenta'))
    # c = load("/home/heygears/work/Tooth_data_prepare/tooth/8AP1S/down.vtk").pointSize(10).c(('green'))
    # show(a, b, c)

    # # convert pts to vtk in train Data
    # pts_path = "/home/heygears/work/Tooth_data_prepare/tooth/pts"
    # stl_path = "/home/heygears/work/Tooth_data_prepare/tooth/stl"
    # vtk_path = "/home/heygears/work/Tooth_data_prepare/tooth/vtk"
    # pts_list = glob.glob(os.path.join(pts_path, "*.pts"))
    # stl_lsit = glob.glob(os.path.join(stl_path, "*.stl"))
    #
    # for pts in pts_list:
    #     file_name = os.path.basename(pts)[:-4]
    #     save_vtk_path = os.path.join(vtk_path, file_name+".vtk")
    #     print("*****", file_name)
    #     gum_line = get_gum_line_pts(pts)
    #     pts_to_vtk(gum_line, save_vtk_path)

    # # show train data
    # vtk_path = "/home/heygears/work/Tooth_data_prepare/tooth/need_change/vtk"
    # stl_path = "/home/heygears/work/Tooth_data_prepare/tooth/need_change/stl"
    # vtk_list = glob.glob(os.path.join(vtk_path, "*.vtk"))
    # stl_list = glob.glob(os.path.join(stl_path, "*.stl"))
    # for i, stl in enumerate(stl_list):
    #     file_name = os.path.basename(stl)[:-4]
    #     vtk = os.path.join(vtk_path, file_name+".vtk")
    #     print(i, " ", file_name)
    #     a = load(stl).c(('magenta'))
    #     b = load(vtk).pointSize(10).c(('green'))
    #     show(a, b)

    # model_path = "/home/heygears/work/Tooth_data_prepare/tooth/file/show"
    # # model_path = "/home/heygears/work/Tooth_data_prepare/deal_data_tools/file/show"
    # file_list = [os.path.join(model_path, file_path) for file_path in os.listdir(model_path)]
    #
    # for i, file in enumerate(file_list):
    #     if not os.path.isdir(file):
    #         continue
    #     if i < 170:
    #         continue
    #     print("{} file path is: {}".format(i, file))
    #     predict1_path = os.path.join(file, "mesh1.obj")
    #     predict2_path = os.path.join(file, "mesh2.obj")
    #     predict_pts = os.path.join(file, "test.vtk")
    #     show_predict(predict1_path, predict2_path, predict_pts)

    # # ---- Test one ----
    # # predict
    # a = load('/home/heygears/work/Tooth_data_prepare/model_test/predict1.obj').c(('blue'))
    # b = load('/home/heygears/work/Tooth_data_prepare/model_test/predict2.obj').c(('magenta'))
    # c = load("/home/heygears/work/Tooth_data_prepare/model_test/predict.vtk").pointSize(10).c(('green'))
    # show(a, b, c)

    # # origin
    # a = load('/home/heygears/work/Tooth_data_prepare/model_test/origin1.obj').c(('blue'))
    # b = load('/home/heygears/work/Tooth_data_prepare/model_test/origin2.obj').c(('magenta'))
    # c = load("/home/heygears/work/Tooth_data_prepare/model_test/origin2.vtk").pointSize(10).c(('green'))  # 边投影
    # d = load("/home/heygears/work/Tooth_data_prepare/model_test/origin1.vtk").pointSize(10).c(('red'))  # 点投影
    # show(a, b, c, d)

    # # ---- Test Batch ----
    # model_path = "/home/heygears/work/Tooth_data_prepare/model_test/predict_mesh_obj_20201210_100/"
    #
    # file_list = [os.path.join(model_path, file_path) for file_path in os.listdir(model_path)]
    #
    # for i, file in enumerate(file_list):
    #     if not os.path.isdir(file):
    #         continue
    #     print("{} file path is: {}".format(i, file))
    #     predict1_path = os.path.join(file, "predict1.obj")
    #     predict2_path = os.path.join(file, "predict2.obj")
    #     predict_pts = os.path.join(file, "predict.vtk")
    #     show_predict(predict1_path, predict2_path, predict_pts)
    #
    #     origin1_path = os.path.join(file, "origin1.obj")
    #     origin2_path = os.path.join(file, "origin2.obj")
    #     origin1_pts = os.path.join(file, "origin1.vtk")
    #     origin2_pts = os.path.join(file, "origin2.vtk")
    #     show_origin(origin1_path, origin2_path, origin1_pts, origin2_pts)

    # # 投影到二维平面
    # generatePreview()

    # # 简化模型
    # simplify()

    # # hgapi
    # stl_handler = hgapi.STLFileHandler()
    # mesh = hgapi.Mesh()
    # file_name = "/home/heygears/work/Tooth_data_prepare/model_test/originMesh/459650_1_L_19.stl"
    # stl_handler.Read(file_name, mesh)
    #
    # mesh.GenerateFaceNormal(bGenerateFaceArea=True)
    # normal = mesh.GetFaceNormal()
    # area = mesh.GetFaceArea()
    # face_areas = np.array(list(area))
    # print(np.argwhere(face_areas == 0))

    # 去掉夹治具
    # model_dir = "/home/heygears/work/Tooth_data_prepare/tooth/need_change/stl/650663/stl/"
    # save_dir = "/home/heygears/work/Tooth_data_prepare/tooth/need_change/stl/650663/healing_stl/"
    # model_list = glob.glob(os.path.join(model_dir, "*.stl"))
    # print(len(model_list))
    # for i, model_path in enumerate(model_list):
    #     model_name = os.path.basename(model_path)
    #     save_path = os.path.join(save_dir, model_name)
    #     healing(model_path, save_path)
    #     print("{} {} is healed".format(i, model_path))
    pass
