import vedo
from vedo import load, show
import torch
import numpy as np

import sys
sys.path.append("/home/heygears/jinhai_zhou/work/package/hgapi")
import hgapi


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
    vtk_point = vedo.Points(gum_line_pts.reshape(-1, 3))
    vedo.write(vtk_point, save_path, binary=False)
    print("vtk file is saved in ", save_path)


def read_seg(seg):
    seg_labels = np.loadtxt(seg, dtype='float64')
    return seg_labels


def read_sseg(sseg_file):
    sseg_labels = read_seg(sseg_file)
    sseg_labels = np.array(sseg_labels > 0, dtype=np.int32)
    return sseg_labels


def pad(input_arr, target_length, val=0, dim=1):
    shp = input_arr.shape
    npad = [(0, 0) for _ in range(len(shp))]
    npad[dim] = (0, target_length - shp[dim])
    return np.pad(input_arr, pad_width=npad, mode='constant', constant_values=val)


# # test torchScript
class MyDecisionGate(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x
        else:
            return -x


class MyCell(torch.nn.Module):
    def __init__(self, dg):
        super(MyCell, self).__init__()
        self.dg = dg
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        x = self.dg(self.linear(x))
        new_h = torch.tanh(x + h)
        return new_h, new_h


class MyRNNLoop(torch.nn.Module):
    def __init__(self, script_gate, x, h):
        super(MyRNNLoop, self).__init__()
        self.cell = torch.jit.trace(MyCell(script_gate), (x, h))

    def forward(self, xs):
        h, y = torch.zeros(3, 4), torch.zeros(3, 4)
        for i in range(xs.size(0)):
            y, h = self.cell(xs[i], h)
        return y, h


class WrapRNN(torch.nn.Module):
    def __init__(self, script_gate):
        super(WrapRNN, self).__init__()
        self.loop = torch.jit.script(MyRNNLoop(script_gate))

    def forward(self, xs):
        y, h = self.loop(xs)
        return torch.relu(y)


if __name__ == "__main__":
    # a = load('/home/heygears/work/Tooth_data_prepare/tooth/8AP1S/mesh1.obj').c(('blue'))
    # b = load('/home/heygears/work/Tooth_data_prepare/tooth/8AP1S/mesh2.obj').c(('magenta'))
    # c = load("/home/heygears/work/Tooth_data_prepare/tooth/8AP1S/down.vtk").pointSize(10).c(('green'))
    # show(a, b, c)

    # # # # convert pts to vtk in train Data
    # pts_path = "/home/heygears/work/Tooth_data_prepare/tooth/three_batch_data/pts"
    # # stl_path = "/run/user/1000/gvfs/smb-share:server=10.99.11.210,share=meshcnn/errorModel-pts"
    # obj_path = "/home/heygears/work/Tooth_data_prepare/tooth/three_batch_data/stl"
    # vtk_path = "/home/heygears/work/Tooth_data_prepare/tooth/three_batch_data/vtk"
    # pts_save_path = "/home/heygears/work/Tooth_data_prepare/tooth/three_batch_data/new_pts"
    # pts_list = glob.glob(os.path.join(pts_path, "*.pts"))
    # # stl_list = glob.glob(os.path.join(stl_path, "*.stl"))
    # obj_list = glob.glob(os.path.join(obj_path, "*.stl"))
    # if not os.path.isdir(vtk_path):
    #     os.makedirs(vtk_path)
    # # if not os.path.isdir(pts_save_path):
    # #     os.makedirs(pts_save_path)
    #
    # for pts in pts_list:
    #     file_name = os.path.basename(pts)[:-4]
    #     save_vtk_path = os.path.join(vtk_path, file_name+".vtk")
    #     save_pts_path = os.path.join(pts_save_path, file_name + ".pts")
    #     print("*****", file_name)
    #     gum_line = get_gum_line_pts(pts)
    #     # np.savetxt(save_pts_path, gum_line)
    #     pts_to_vtk(gum_line, save_vtk_path)
    #
    # # show train data
    # vtk_list = glob.glob(os.path.join(vtk_path, "*.vtk"))
    # # for i, stl in enumerate(stl_list):
    # for i, stl in enumerate(obj_list):
    #         file_name = os.path.basename(stl)[:-4]
    #         vtk = os.path.join(vtk_path, file_name+".vtk")
    #         print(i+1, " ", file_name)
    #         a = load(stl).c(('magenta'))
    #         b = load(vtk).pointSize(10).c(('green'))
    #         show(a, b)

    # model_path = "/home/heygears/work/Tooth_data_prepare/tooth/three_batch_data/file/show"
    # # model_path = "/home/heygears/work/Tooth_data_prepare/deal_data_tools/file/show"
    # file_list = [os.path.join(model_path, file_path) for file_path in os.listdir(model_path)]
    #
    # for i, file in enumerate(file_list):
    #     if not os.path.isdir(file):
    #         continue
    #     print("{} file path is: {}".format(i, file))
    #     predict1_path = os.path.join(file, "mesh1.obj")
    #     predict2_path = os.path.join(file, "mesh2.obj")
    #     predict_pts = os.path.join(file, "test.vtk")
    #     show_predict(predict1_path, predict2_path, predict_pts)

    # ---- Test one ----
    # predict
    # a = load('/home/heygears/work/Tooth_data_prepare/deal_data_tools/predict1.obj').c(('blue'))
    # b = load('/home/heygears/work/Tooth_data_prepare/deal_data_tools/predict2.obj').c(('magenta'))
    # c = load("/home/heygears/work/Tooth_data_prepare/deal_data_tools/predict.vtk").pointSize(10).c(('green'))
    # show(a, b, c)

    # # origin
    # a = load('/home/heygears/work/Tooth_data_prepare/model_test/origin1.obj').c(('blue'))
    # b = load('/home/heygears/work/Tooth_data_prepare/model_test/origin2.obj').c(('magenta'))
    # c = load("/home/heygears/work/Tooth_data_prepare/model_test/origin2.vtk").pointSize(10).c(('green'))  # 边投影
    # d = load("/home/heygears/work/Tooth_data_prepare/model_test/origin1.vtk").pointSize(10).c(('red'))  # 点投影
    # show(a, b, c, d)

    # # ---- Test Batch ----
    # model_path = "/home/heygears/work/Tooth_data_prepare/model_test/predict_mesh_obj_20201216_200/"
    #
    # file_list = [os.path.join(model_path, file_path) for file_path in os.listdir(model_path)]
    #
    # j = 1
    # for i, file in enumerate(file_list):
    #     if not os.path.isdir(file):
    #         continue
    #     print("{} file path is: {}".format(j, file))
    #     predict1_path = os.path.join(file, "predict1.obj")
    #     predict2_path = os.path.join(file, "predict2.obj")
    #     predict_pts = os.path.join(file, "predict.vtk")
    #     show_predict(predict1_path, predict2_path, predict_pts)
    #     j += 1
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

    # # # add loss
    # seg_file = "/home/heygears/work/github/MeshCNN/datasets/tooth_seg/seg/1PT7E_VS_SET_VSc2_Subsetup_Retainer_Maxillar.eseg"
    # label = read_sseg(seg_file) - 1
    # label = pad(label, 7500, val=-1, dim=0)
    # labels = torch.from_numpy(label[np.newaxis, :]).long()
    # print("labels: ", labels, labels.size())
    # #
    # out = np.loadtxt("./out.txt")
    # out = torch.from_numpy(out[np.newaxis, :])
    # print("out: ", out, out.size())
    # classes = out.data.max(1)[1]
    # print("classes: ", classes, classes.size())
    #
    # criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    # loss = criterion(out, labels)
    # print(loss)

    # soft_label = np.loadtxt("./soft_label.txt")
    # print(soft_label.shape, soft_label[40:50])

    # sseg_file = "/home/heygears/work/github/MeshCNN/datasets/tooth_seg/sseg/1PT7E_VS_SET_VSc2_Subsetup_Retainer_Maxillar.seseg"
    # sseg_label = np.loadtxt(sseg_file, dtype="float64")
    # # print(sseg_label[40:50])
    #
    # soft_label = read_sseg(sseg_file)
    # soft_label = torch.from_numpy(soft_label)
    # print("soft_label: ", soft_label.size())
    # # print(soft_label[:, 0][40:50])
    # # print(soft_label[:, 1][40:50])
    # border_edge_ids = torch.where(((soft_label[:, 0] == 1) & (soft_label[:, 1] == 1)))[0]
    # # border_edge_ids = border_edge_ids.reshape((1, 596))
    # out_edges = []
    # out = out.squeeze()
    # out_edges = out[:, border_edge_ids].unsqueeze(0)
    # print(out_edges)
    # for i in range(len(out)):
    #     out_edge = out[i][border_edge_ids]
    #     out_edges.append(out_edge.numpy())
    # out_edges = torch.tensor(out_edges).unsqueeze(0)
    # print(out_edges.size(), out_edges)

    # label_edges = labels[0, border_edge_ids].unsqueeze(0)
    # print(label_edges.size())
    #
    # criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    # loss = criterion(out_edges, label_edges)
    # print(loss)

    # # # 从共享文件夹中提取数据
    import os
    import shutil
    print("hello")
    import vedo
    import numpy as np
    gum_line_pts = np.array([1, 2, 3, 4, 5, 6])
    save_path = "./test.vtk"
    vtk_point = vedo.Points(gum_line_pts.reshape(-1, 3))
    vedo.write(vtk_point, save_path, binary=False)
    print("vtk file is saved in ", save_path)
    print("*****")
    p = vedo.load("./test.vtk").pointSize(10).c(('green'))
    vedo.show(p)
    # stl_save_dir = "/run/user/1000/gvfs/smb-share:server=10.99.11.210,share=meshcnn/Test_5044/stl"
    # error_txt = "/run/user/1000/gvfs/smb-share:server=10.99.11.210,share=meshcnn/Test_5044/error_label1.txt"
    # error_dir = "/run/user/1000/gvfs/smb-share:server=10.99.11.210,share=meshcnn/Test_5044/error_stl"
    #
    # with open(error_txt, "r") as f:
    #     for line in f:
    #         line = line.strip()
    #         stl_file = os.path.join(stl_save_dir, line + ".stl")
    #         if os.path.isfile(stl_file):
    #             shutil.copyfile(stl_file, os.path.join(error_dir, line + ".stl"))
    #             print(stl_file)
    #
    # error_model_stl = glob.glob(os.path.join(error_dir, "*.obj"))
    # print(len(error_model_stl))
    # for obj_model in error_model_stl:
    #     basename = os.path.basename(obj_model)
    #     file_name = os.path.splitext(basename)[0] + ".stl"
    #     file_path = os.path.join(stl_dir, file_name)
    #     shutil.copyfile(file_path, os.path.join(stl_save_dir, file_name))
    #
    # obj_dir = "/home/heygears/work/Tooth_data_prepare/tooth/error_test/test"
    # save_dir = "/home/heygears/work/Tooth_data_prepare/tooth/error_test/error/obj"
    # stl_dir = "/run/user/1000/gvfs/smb-share:server=10.99.11.210,share=meshcnn/errorModel-pts"
    # stl_save_dir = "/home/heygears/work/Tooth_data_prepare/tooth/error_test/error/pts"
    # error_model_file = "./error_model.txt"
    #
    # with open(error_model_file, "r") as f:
    #     for line in f:
    #         line = line.strip()
    #         obj_name = line + ".obj"
    #         obj_path = os.path.join(obj_dir, obj_name)
    #         if os.path.isfile(obj_path):
    #             shutil.move(obj_path, save_dir)
    #             stl_path = os.path.join(stl_dir, line + ".pts")
    #             print(stl_path)
    #             if os.path.isfile(stl_path):
    #                 shutil.copyfile(stl_path, os.path.join(stl_save_dir, line+".pts"))

    # stl_dir = "/home/heygears/work/Tooth_data_prepare/tooth/three_batch_data/error/stl"
    # train_dir = "/home/heygears/work/Tooth_data_prepare/tooth/three_batch_data/error/file/train"
    # test_dir = "/home/heygears/work/Tooth_data_prepare/tooth/error_test/test"
    # no_seg_dir = "/home/heygears/work/Tooth_data_prepare/tooth/error_test/stl"
    # stl_lists = glob.glob(os.path.join(stl_dir, "*.stl"))
    # print(len(stl_lists))
    # obj_lists = glob.glob(os.path.join(train_dir, "*.obj"))
    # print(len(obj_lists))
    #
    # for stl in stl_lists:
    #     base_name = os.path.basename(stl)
    #     obj_name = os.path.splitext(base_name)[0] + ".obj"
    #     train_obj_path = os.path.join(train_dir, obj_name)
    #
    #     if os.path.isfile(train_obj_path):
    #         continue
    #
    #     test_obj_path = os.path.join(test_dir, obj_name)
    #
    #     no_seg_stl_path = os.path.join(stl_dir, base_name)
    #     print(no_seg_stl_path)
    #     if os.path.isfile(no_seg_stl_path) and not os.path.isfile(os.path.join(no_seg_dir, base_name)):
    #         shutil.move(no_seg_stl_path, no_seg_dir)

    # # test torchScript
    #
    # my_cell = MyCell(MyDecisionGate())
    # x = torch.rand(3, 4)
    # h = torch.rand(3, 4)
    # #
    # # trace
    # traced_cell = torch.jit.trace(my_cell, (x, h))
    # print("trace code: \n", traced_cell.code)
    #
    # # script
    # script_gate = torch.jit.script(MyDecisionGate())
    # my_cell = MyCell(script_gate)
    # script_cell = torch.jit.script(my_cell)
    # print("script code: \n", script_cell.code)
    #
    # # mixing scripting and tracing
    # rnn_loop = torch.jit.script(MyRNNLoop(script_gate))
    # print("rnn_loop code: \n", rnn_loop.code)
    #
    # traced = torch.jit.trace(WrapRNN(), (torch.rand(1, 3, 4)))
    # print(traced.code)
    # traced.save("./wrapped_rnn.zip")
    # loaded = torch.jit.load("./wrapped_rnn.zip")
    # print(loaded)
    # print(loaded.code)
    # z = torch.rand(1, 3, 4)
    # print(loaded(z))

    # torch.jit.save(traced, "./test.pt")
    # loaded = torch.jit.load("./test.pt")
    # print(loaded(z))

    # @torch.jit.script
    # class Mesh(object):
    #     def __init__(self):
    #         self.gemm_edges = None
    #         self.sides = None
    #         self.faces = None
    #         self.pool_count = 1
    #
    # class MyModule(torch.nn.Module):
    #     def __init__(self, mesh):
    #         super(MyModule, self).__init__()
    #         self.mesh = mesh
    #
    #     def forward(self, x):
    #         out = x + self.mesh.pool_count
    #         return out
    #
    #
    # mesh = Mesh()
    # m_1 = torch.jit.script(MyModule(mesh))
    # print(m_1.code)
    #
    # print(mesh.pool_count)
    # model = MyModule(mesh)
    # print(model(torch.tensor(3)))

    # from typing import Tuple
    # @torch.jit.script
    # def foo(x: int, tup: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    #     t0, t1 = tup
    #     return t0 + t1 + x
    # m = foo(3, (torch.tensor(2), torch.tensor(1)))
    # print(foo.code, "\n", m)

    from typing import Dict, List, Tuple

    # class EmptyDataStructures(torch.nn.Module):
    #     def __init__(self):
    #         super(EmptyDataStructures, self).__init__()
    #
    #     def forward(self, x):
    #         # This annotates the list to be a `List[Tuple[int, float]]`
    #         my_list: List[Tuple[int, float]] = []
    #         for i in range(10):
    #             my_list.append((i, x.item()))
    #
    #         my_dict: Dict[str, int] = {}
    #         return my_list, my_dict
    #
    #
    # x = torch.jit.script(EmptyDataStructures())
    # print(x.code)

    # from typing import Optional
    #
    # class M(torch.nn.Module):
    #     z: Optional[int]
    #
    #     def __init__(self, z):
    #         super(M, self).__init__()
    #         # If `z` is None, its type cannot be inferred, so it must
    #         # be specified (above)
    #         self.z = z
    #
    #     def forward(self, x, y, z):
    #         # type: (Optional[int], Optional[int], Optional[int]) -> int
    #         if x is None:
    #             x = 1
    #             x = x + 1
    #
    #         # Refinement for an attribute by assigning it to a local
    #         z = self.z
    #         if y is not None and z is not None:
    #             x = y + z
    #
    #         # Refinement via an `assert`
    #         assert z is not None
    #         x += z
    #         return x
    #
    #
    # module = torch.jit.script(M(2))
    # print(module.code)
    # module = torch.jit.script(M(None))
    # print(module.code)

    # import glob
    # import os
    # import shutil
    #
    # correct_dir = "/run/user/1000/gvfs/smb-share:server=10.99.11.210,share=meshcnn/Test_5044/error/modify_correct/5"
    # with open("./error_model.txt", "r") as f:
    #     for line in f:
    #         line = line.strip()
    #         file_path = os.path.join(correct_dir, line+".pts")
    #         if not os.path.isfile(file_path):
    #             print(file_path)

    pass
