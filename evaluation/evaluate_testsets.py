#!/usr/bin/env python
# coding: utf-8

# ### step1 读取pts文件 获取标注点集$A$

# In[1]:


import os
import numpy as np
from scipy.spatial import KDTree
import glob
import math
import sys
import vedo


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


def get_target_pts(pts_path):
    """
    获取真实的牙龈线点
    """
    target_pts = np.loadtxt(pts_path)
    return target_pts


# ### step2 对点集A建立KD_Tree

# In[2]:


def create_tree(array):
    """
    根据点集建立kd_tree
    """
    tree = KDTree(array)
    return tree


# ### step3 预测结果生成对应的点集$B$  
# 边标签--->点标签


# In[3]:
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
                    faces.append([int(c.split('/')[0]) - 1 for c in splitted_line[1:]])
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
    # if len(edges) == 0:
    #     edges = get_edges(faces)
    edges = np.array(edges)

    return vs, faces, edges


def label_face_by_edge(faces, edges, edge_labels):
    """
    利用边标签对面进行标记
    @faces: 模型的面
    @edges: 模型的边
    @edge_labels: 模型边对应的标签
    return: 面的标签
    """
    edge_dict = {}  # key: str([pt1, pt2]) value: label
    for ei, edge in enumerate(edges):
        key = tuple(edge)
        edge_dict[key] = edge_labels[ei]
    # print(edge_dict)
    face_labels = np.array(len(faces) * [[-1, -1, -1]])
    for i, face in enumerate(faces):
        for j in range(3):
            cur_edge = [face[j], face[(j + 1) % 3]]
            cur_label = edge_dict[tuple(sorted(cur_edge))]
            face_labels[i][j] = cur_label

    face_labels = np.where(np.sum(face_labels, axis=1) < 2, 1, 2)

    optimizer_face_labels(faces, face_labels)  # 对面标签进行优化  膨胀操作 填充

    return face_labels


def find_neighb_faces(face_id, faces):
    face = faces[face_id]
    nb_face = []
    for i in range(3):
        cur_edge = [face[i], face[(i + 1) % 3]]
        pt1 = cur_edge[0]
        pt2 = cur_edge[1]
        face_ids = find_faces_by_2point(faces, pt1, pt2)
        if len(face_ids) == 2:
            nb_face_id = face_ids[0][0] if face_ids[0][0] != face_id else face_ids[1][0]
            nb_face.append(nb_face_id)
    return nb_face


def optimizer_face_labels(faces, face_labels):
    # new_face_labels = face_labels.copy()
    for i, face in enumerate(faces):
        nb_faces = find_neighb_faces(i, faces)
        nb_labels = []
        for face_id in nb_faces:
            nb_labels.append(face_labels[face_id])
        if len(nb_labels) == 0:
            continue
        counts = np.bincount(nb_labels)
        # 返回众数
        if face_labels[i] != np.argmax(counts):
            # print("face: {}, label:{} nb_labels: {}, 众数: {}".format(i, face_labels[i], nb_labels, np.argmax(counts)))
            face_labels[i] = np.argmax(counts)


def find_faces_by_2point(faces, id1, id2):
    """
    根据两个点确定以两点所在边为公共边的两个面
    @faces: 所有面，N*3, 值表示点的id值
    @id1: 第一个点的id值
    @id2: 第二个点的id值
    return: 2*3, [面的id，第一个点的位置， 第二个点的位置]
    """
    p1_faces = np.argwhere(faces == id1)  # 行id, 列id
    p2_faces = np.argwhere(faces == id2)

    intersection_faces = []
    for val1 in p1_faces:
        for val2 in p2_faces:
            if val1[0] == val2[0]:
                intersection_faces.append([val1[0], val1[1], val2[1]])

    return intersection_faces


def get_pts_from_edges(edges, threshold=30):
    circle_pts = [[]]
    count = 0
    while len(edges) > 0:
        if len(circle_pts[count]) == 0:
            circle_pts[count] = list(edges[0])
            edges = np.delete(edges, 0, axis=0)
        else:
            last_id = circle_pts[count][-1]
            idx = np.where(edges == last_id)[0]
            if len(idx) == 0:
                circle_pts.append([])
                count += 1
            else:
                edge = edges[idx[0]]
                next_id = edge[0] if edge[0] != last_id else edge[1]
                circle_pts[count].append(next_id)
                edges = np.delete(edges, idx[0], axis=0)

    pts_ids = []
    for circle in circle_pts:
        # 过滤短的
        if len(circle) > threshold:
            # print("{}".format(len(circle)))
            circle = drop_cycle(circle, threshold)   # 去闭环
            # print("after drop cycle {}".format(len(circle)))
            pts_ids.append(circle)

    # TODO 如果len(pts_ids) > 0 需要合并
    return pts_ids


def drop_cycle(edge, max_length=20):
    """
    删除列表中形成的小闭环
    @edge: 原始顶点id
    @max_length: 容许闭环的最小长度
    return: 输出删除小闭环后的列表
    """
    drop_list = []
    drop_count = 0
    for i, item in enumerate(edge):
        if item not in drop_list:
            drop_list.append(item)
        else:
            last_index = len(drop_list) - 1 - drop_list[::-1].index(item)
            if i - last_index - drop_count < max_length:
                drop_count += len(drop_list[last_index:])
                drop_list = drop_list[:last_index+1]
            else:
                drop_list.append(item)

    # 去掉首尾构成的闭环  如： [956 1035 1538 ...... 2028 1035 952 956] ==> 1035->952->956->1035
    circle_count = np.where(np.bincount(drop_list) >= 2)[0]
    for item in circle_count:
        if item == drop_list[0]:
            continue
        first_id = drop_list.index(item)
        last_id = drop_list[::-1].index(item)
        if first_id + last_id <= max_length:
            length = len(drop_list)
            drop_list = drop_list[first_id:length-last_id]

    return np.asarray(drop_list)

# def label_pts_by_edges_and_faces(vs, edges, faces, face_labels):
#     """
#     根据边和面标签，对点进行标注，一条边对应两个面，如果两个面标签不同，则保留点
#     @vs: 模型的点
#     @edges: 模型的边
#     @faces: 模型的面
#     @face_labels: 模型面对应的标签
#     return: 模型边界点
#     """
#     pts_labels = np.array(len(vs) * [False])
#     for ei, edge in enumerate(edges):
#         pt1 = edge[0]
#         pt2 = edge[1]
#         face_ids = find_faces_by_2point(faces, pt1, pt2)
#         if len(face_ids) == 2:
#             if face_labels[face_ids[0][0]] != face_labels[face_ids[1][0]]:
#                 pts_labels[pt1] = True
#                 pts_labels[pt2] = True
#
#     return vs[pts_labels]


def label_pts_by_edges_and_faces(vs, edges, faces, face_labels):
    """
    根据边和面标签，对点进行标注，一条边对应两个面，如果两个面标签不同，则保留点
    @vs: 模型的点
    @edges: 模型的边
    @faces: 模型的面
    @face_labels: 模型面对应的标签
    return: 模型边界点
    """
    edge_idx = []
    for ei, edge in enumerate(edges):
        pt1 = edge[0]
        pt2 = edge[1]
        face_ids = find_faces_by_2point(faces, pt1, pt2)
        if len(face_ids) == 2:
            if face_labels[face_ids[0][0]] != face_labels[face_ids[1][0]]:
                edge_idx.append(ei)
    test_edges = np.asarray(edges[edge_idx])
    pts_ids = get_pts_from_edges(test_edges)
    # TODO 如果len(pts_ids) > 0 需要合并(多个闭环)
    idx = np.array([], dtype=int)
    face_normals, face_areas = compute_face_normals_and_areas(vs, faces)  # 计算面的法向量
    for pts_id in pts_ids:
        # idx = np.append(idx, pts_id)
        temp = []
        temp.append(pts_id[0])
        for i in range(1, len(pts_id) - 1):
            last_pt = pts_id[i - 1]
            cur_pt = pts_id[i]
            next_pt = pts_id[i + 1]
            a = vs[last_pt] - vs[cur_pt]
            b = vs[next_pt] - vs[cur_pt]
            y = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
            x = math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) * math.sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2])
            # 计算三点形成的夹角
            theta = math.acos(y / x) / math.pi * 180  # 不存在重合点 所以x可以不用判零
            if theta > 50:
                curvature = compute_pt_curvature(vs, edges, faces, face_normals, cur_pt)
                if max(curvature) > 0:
                    temp.append(cur_pt)
        temp.append(pts_id[-1])
        idx = np.append(idx, temp)

    return vs[idx]


def compute_face_normals_and_areas(vs, faces):
    """
    计算每个面的法向量和面积
    """
    face_normals = np.cross(vs[faces[:, 1]] - vs[faces[:, 0]],
                            vs[faces[:, 2]] - vs[faces[:, 1]])

    # >>> deal zero face >>>
    zeros_idx = np.argwhere((face_normals[:, 0] == 0) & (face_normals[:, 1] == 0) & (face_normals[:, 2] == 0))
    normal_mean = np.mean(face_normals, axis=0)
    for idx in zeros_idx:
        idx = idx[0]
        face_normals[idx] = normal_mean
        # print("face_normals_idx: ", face_normals[idx])
    # <<< deal zero face <<<
    face_areas = np.sqrt((face_normals ** 2).sum(axis=1))
    # print("n_faces: ", len(faces), mesh.filename)
    face_normals /= face_areas[:, np.newaxis]
    assert (not np.any(face_areas[:, np.newaxis] == 0)), "has zero area face!"
    face_areas *= 0.5
    return face_normals, face_areas


def compute_pt_curvature(vs, edges, faces, face_normals, pt_id):
    # Reference: https://doi.org/10.1145/3394486.3403272 CurvaNet
    c_ij = []
    edge_ids, cur_idxs = np.where(edges == pt_id)
    for i, edge_id in enumerate(edge_ids):
        cur_pt_id = cur_idxs[i]
        point_i = edges[edge_id][cur_pt_id]
        point_j = edges[edge_id][1 - cur_pt_id]
        normal_i = compute_point_normal(faces, face_normals, point_i)
        e_ij = vs[point_j] - vs[point_i]
        c_ij.append(2 * normal_i.dot(e_ij / (np.sqrt((e_ij ** 2).sum()) + sys.float_info.epsilon)))

    return c_ij


def compute_point_normal(faces, face_normals, point_id):
    face_ids = get_faces_by_point(faces, point_id)
    normal_sum = face_normals[face_ids].sum(0)   # 按行相加
    normal_div = np.sqrt((normal_sum ** 2).sum())
    normal = normal_sum / (normal_div + sys.float_info.epsilon)
    return normal


def get_faces_by_point(faces, point_id):
    point_faces = np.argwhere(faces == point_id)
    face_ids = point_faces[:, 0]
    return face_ids


# In[4]:
# ### 6.分开保存模型 便于显示
# In[9]:
def save_model_part(save_path, vs, faces, face_labels, model1_name="mesh1.obj", model2_name="mesh2.obj"):
    """
    根据标签将模型标记的部分分别保存
    @obj_vs: 模型的顶点
    @obj_faces: 模型的面
    @face_labels: 面的标签 
    return: None
    """
    mesh1 = open(os.path.join(save_path, model1_name), "w")
    mesh2 = open(os.path.join(save_path, model2_name), "w")
    for v in vs:
        mesh1.write("v " + str(v[0]) + " " + str(v[1]) + " " + str(v[2]) + "\n")
        mesh2.write("v " + str(v[0]) + " " + str(v[1]) + " " + str(v[2]) + "\n")

    for idx, face in enumerate(faces):
        if face_labels[idx] == 1:
            mesh1.write("f " + str(face[0] + 1) + " " + str(face[1] + 1) + " " + str(face[2] + 1) + "\n")
        if face_labels[idx] == 2:
            mesh2.write("f " + str(face[0] + 1) + " " + str(face[1] + 1) + " " + str(face[2] + 1) + "\n")

    mesh1.close()
    mesh2.close()


# ### 7. 导出边界点

# In[10]:


def save_pts_to_vtk(pts, save_path="./test.vtk"):
    """
    将牙龈点pts格式转为vtk格式
    @pts: 点集 [[x, y, z], [x, y, z], ...]
    @save_path: 保存路径
    return: None
    """
    vtk_point = vedo.Points(pts.reshape(-1, 3))
    vedo.write(vtk_point, save_path, binary=False)
#     print("vtk file is saved in ", save_path)


# In[5]:


def get_predict_pts(model_path):
    predict_vs, predict_faces, predict_edges = parse_obje(model_path)

    if len(predict_edges) == 0:
        print("{} is no result!".format(model_path))
        return np.array([])

    origin_model_basename = os.path.basename(model_path)[:-6]
    predict_path = os.path.split(model_path)[0]
    save_path = os.path.join(predict_path, origin_model_basename)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    predict_labels = predict_edges[:, -1]
    predict_edges = predict_edges[:, :-1]

    # ## 标记预测的面
    predict_face_labels = label_face_by_edge(predict_faces, predict_edges, predict_labels)
    save_model_part(save_path, predict_vs, predict_faces, predict_face_labels, "predict1.obj", "predict2.obj")

    # ## 通过面的标签来判断
    predict_gum_pts = label_pts_by_edges_and_faces(predict_vs, predict_edges, predict_faces, predict_face_labels)
    # print("predict_gum_pts: ", len(predict_gum_pts))
    np.savetxt(os.path.join(save_path, "predict.pts"), predict_gum_pts)
    save_pts_to_vtk(predict_gum_pts, os.path.join(save_path, "predict.vtk"))

    return predict_gum_pts


# ### step4 遍历点集$B$ 找出点到集合$A$最大距离

# In[6]:


def calculate_dist(tree, predict_pts):
    max_dist = 0
    max_index = 0
    for idx, predict_pt in enumerate(predict_pts):
        dist, index = tree.query(predict_pt, k=1)
        if dist > max_dist:
            max_dist = dist
            max_index = idx

    return max_dist, max_index


# In[7]:


def get_max_dist(pts_file, predict_model):
    target_pts = get_gum_line_pts(pts_file)

    tree = create_tree(target_pts)

    # TODO 可以替换成其他方法得到的分割点
    model_name = os.path.splitext(os.path.basename(predict_model))[0][:-2]
    save_dir = os.path.split(predict_model)[0]
    predict_save_path = os.path.join(save_dir, model_name + "/predict.pts")
    if os.path.isfile(predict_save_path):
        predict_pts = np.loadtxt(predict_save_path)
    else:
        predict_pts = get_predict_pts(predict_model)

    if len(predict_pts) == 0:
        return np.inf, np.inf   # 无结果

    max_dist, max_index = calculate_dist(tree, predict_pts)

    return max_dist, predict_pts[max_index]


# In[8]:
def show_predict(predict1, predict2, pts, target_pts, max_dist_pts=None):
    from vedo import load, show, Point
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
    t = load(target_pts).pointSize(5).c(("red"))
    if max_dist_pts:
        p1 = Point(max_dist_pts, r=15, c='yellow')
        show(a, b, c, t, p1)
    else:
        show(a, b, c, t)


# In[12]:
if __name__ == "__main__":
    pts_dir = "/data/GumLine_Data_5044/pts/"
    predict_model_dir = "/home/heygears/jinhai_zhou/work/predict_results/"
    threshold = 2

    model_paths = glob.glob(os.path.join(predict_model_dir, "*.obj"))
    error_models = dict()

    for idx, model_path in enumerate(model_paths):
        model_name = os.path.splitext(os.path.basename(model_path))[0][:-2]
        pts_file = os.path.join(pts_dir, model_name + ".pts")
        if os.path.isfile(pts_file):
            max_dist, max_pts = get_max_dist(pts_file, model_path)
            if max_dist is np.inf and max_pts is np.inf:
                continue
            if max_dist > threshold:
                error_models[os.path.splitext(os.path.basename(pts_file))[0]] = [max_dist, max_pts.tolist()]
            print("{}, file: {}, dist: {}".format(idx + 1, os.path.basename(pts_file), max_dist))

    print("******一共{}个模型的距离超过阈值{}，占总模型数{}的百分比为：{}*********\n".
          format(len(error_models), threshold, len(model_paths), len(error_models) / len(model_paths)))

    # save error_models
    with open("error_label.txt", "w") as f:
        for key in error_models.keys():
            f.write(key)
            f.write("\n")

    # show model  或者保存 error_models
    file_list = [os.path.join(predict_model_dir, file_path) for file_path in error_models.keys()]

    for i, file in enumerate(file_list):
        if os.path.basename(file) in error_models:
            print("{} file path is: {}, dist: {}".format(i + 1, file, error_models[os.path.basename(file)][0]))
            predict1_path = os.path.join(file, "predict1.obj")
            predict2_path = os.path.join(file, "predict2.obj")
            predict_pts = os.path.join(file, "predict.vtk")
            # >>> target vtk
            model_name = os.path.basename(file)
            pts_file = os.path.join(pts_dir, model_name + ".pts")
            target_pts = get_gum_line_pts(pts_file)
            target_pts_save_path = os.path.join(file, "target.vtk")
            save_pts_to_vtk(target_pts, target_pts_save_path)
            # <<< target vtk
            show_predict(predict1_path, predict2_path, predict_pts, target_pts_save_path, error_models[os.path.basename(file)][1])




