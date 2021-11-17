#!/usr/bin/env python
# coding: utf-8
""""
Usage: python show_data.py
"""

# In[1]:


import os 
import numpy as np
from scipy import spatial
import glob
from multiprocessing import Process
from tqdm import tqdm
from vedo import load, show, Point
import math
import sys


# ## 一、自定义函数

# ### 1.获取模型信息

# In[2]:


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
                    faces.append([int(c.split('/')[0]) - 1 for c in splitted_line[1:]])                   
            elif splitted_line[0] == 'e':
                if len(splitted_line) >= 4:
                    edge_v = [int(c) - 1 for c in splitted_line[1:-1]]
                    edge_c = int(splitted_line[-1])
                    edge_v.append(edge_c)                 # class
                    edges.append(edge_v)           
            else:
                continue

    vs = np.array(vs)
    faces = np.array(faces, dtype=int)
    # if len(edges) == 0:
    #     edges = get_edges(faces)
    edges = np.array(edges)
        
    return vs, faces, edges


# ### 2.根据边标记对面进行标记

# In[3]:


def label_face_by_edge(faces, edges, edge_labels):
    """
    利用边标签对面进行标记
    @faces: 模型的面
    @edges: 模型的边
    @edge_labels: 模型边对应的标签
    return: 面的标签
    """
    edge_dict = {}    # key: str([pt1, pt2]) value: label
    for ei, edge in enumerate(edges):
        key = tuple(edge)
        edge_dict[key] = edge_labels[ei]
    # print(edge_dict)
    face_labels = np.array(len(faces) * [[-1, -1, -1]])
    for i, face in enumerate(faces):
        # faces_edges = []
        for j in range(3):
            cur_edge = [face[j], face[(j + 1) % 3]]
            cur_label = edge_dict[tuple(sorted(cur_edge))]
            face_labels[i][j] = cur_label
            
        # face_labels.append(faces_edges)
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

# ### 3.利用边对点进行标记

# In[4]:


def label_pts_by_edges(vs, edges, edge_labels):
    """
    根据边标签，对点进行标注
    @vs: 模型的点
    @edge: 模型的边
    @edge_labels: 模型边对应的标签
    return: 模型点的标签
    """
    pts_labels = np.array(len(vs) * [[-1, -1]])
    for ei, edge in enumerate(edges):
        edge_label = edge_labels[ei]
        pt1 = edge[0]
        pt2 = edge[1]
        pts_labels[pt1][edge_label] = edge_label
        pts_labels[pt2][edge_label] = edge_label
    
    return pts_labels


# In[5]:
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


# In[6]:
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
    # pts_labels = np.array(len(vs) * [False])
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


# ### 4.边标签投影到原始模型

# In[7]:
def label_origin_edge(predict_edges, predict_labels, predict_vs, origin_edges, origin_vs):
    """
    根据预测的边及标签，对原始模型的边进行标注
    @predict_edges: 预测模型对应的边
    @predict_labels: 预测模型对应的标签
    @origin_edges: 原始模型的边
    return: 原始模型边对应的标签
    """
    predict_edge_pts = predict_vs[predict_edges].reshape(-1, 6)

    tree = spatial.KDTree(predict_edge_pts)

    origin_edge_pts = origin_vs[origin_edges].reshape(-1, 6)

    origin_labels = []
    for i, edge in enumerate(origin_edge_pts):
        # if i % 50000 == 0:
        #     print(i, "is finded!")
        dist, idx = tree.query(edge)
        origin_labels.append(predict_labels[idx])
    
    return origin_labels


# ### 5.点投影到原模型

# In[8]:


def project_points(predict_pts, origin_vs):
    """
    根据预测的边，筛选出边界点，将点投影回原模型
    @predict_pts: 边界点
    @origin_vs: 原始模型所有点
    return: 返回原始模型的边界点
    """
    tree = spatial.KDTree(origin_vs)
    
    origin_pts = []
    for i, pt in enumerate(predict_pts):
        dist, idx = tree.query(pt)
        origin_pts.append(origin_vs[idx])
        
    origin_pts = np.asarray(origin_pts)
    return origin_pts
    

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
            mesh1.write("f " + str(face[0]+1) + " " + str(face[1]+1) + " " + str(face[2]+1) + "\n")
        if face_labels[idx] == 2:
            mesh2.write("f " + str(face[0]+1) + " " + str(face[1]+1) + " " + str(face[2]+1) + "\n")

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
    import vtkplotter as vtkp
    vtk_point = vtkp.Points(pts.reshape(-1, 3))
    vtkp.write(vtk_point, save_path, binary=False)
#     print("vtk file is saved in ", save_path)


# ## 二、主函数
# 

# In[12]:


def save_predict(predict_model, predict_path):
    """
    对预测的模型进行分析，找出牙龈线点，并对原始模型进行分割
    @predict_model: 预测的模型
    @save_path: 结果保存路径
    return: None
    """
    # ------加载模型 获取信息------
    # ## 预测模型
    predict_vs, predict_faces, predict_edges = parse_obje(predict_model)
    if len(predict_edges) == 0:
        print("{} is no result!".format(predict_model))
        return

    origin_model_basename = os.path.basename(predict_model)[:-6]
    save_path = os.path.join(predict_path, origin_model_basename)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    predict_labels = predict_edges[:, -1]
    predict_edges = predict_edges[:, :-1]
    
    # ## 标记预测的面
    predict_face_labels = label_face_by_edge(predict_faces, predict_edges, predict_labels)

    save_model_part(save_path, predict_vs, predict_faces, predict_face_labels, "predict1.obj", "predict2.obj")
    
    # ------处理预测模型------
    # # 方案一 直接通过边解析点
    # predict_pts_labels = label_pts_by_edges(predict_vs, predict_edges, predict_labels)
    # predict_gum_pt_ids = np.where((predict_pts_labels[:,0]==0) & (predict_pts_labels[:,1]==1))[0]
    # predict_gum_pts = predict_vs[predict_gum_pt_ids]
    # print("predict_gum_pts: ", len(predict_gum_pts))
    # save_pts_to_vtk(predict_gum_pts, os.path.join(save_path, "predict.vtk"))
    
    # ## 方案二 通过面的标签来判断
    predict_gum_pts = label_pts_by_edges_and_faces(predict_vs, predict_edges, predict_faces, predict_face_labels)
    # print("predict_gum_pts: ", len(predict_gum_pts))
    np.savetxt(os.path.join(save_path, "predict.pts"), predict_gum_pts)
    save_pts_to_vtk(predict_gum_pts, os.path.join(save_path, "predict.vtk"))


# ## 三、批量处理

# In[15]:


def show_predict_batch(predict_model_list, predict_path):
    """
    批量处理预测模型
    @predict_model_list: 预测的模型列表
    @predict_path: 预测模型存放路径
    return: None
    """
    for i, predict_model in enumerate(tqdm(predict_model_list)):
        try:
            save_predict(predict_model, predict_path)
        except KeyError:
            print("predict_model: ", predict_model)
        except Exception as e:
            raise e


# In[16]:


def parallel_show_predict(model_list, predict_path, n_workers=8):
    """
    多进程处理
    """
    if len(model_list) < n_workers:
        n_workers = len(model_list)
    chunk_len = len(model_list) // n_workers

    chunk_lists = [model_list[i:i+chunk_len] for i in range(0, (n_workers-1)*chunk_len, chunk_len)]
    chunk_lists.append(model_list[(n_workers - 1)*chunk_len:])
    
    process_list = [Process(target=show_predict_batch, args=(chunk_list, predict_path, )) for chunk_list in chunk_lists]
    for process in process_list:
        process.start()
    for process in process_list:
        process.join()


# In[17]:
def show_predict(predict1, predict2, pts, max_dist_pts=None):
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
    if max_dist_pts:
        p1 = Point(max_dist_pts, r=20, c='yellow')
        show(a, b, c, p1)
    else:
        show(a, b, c)


if __name__ == "__main__":

    predict_dir = "/home/heygears/work/predict_results"

    # 解析结果
    predict_model_list = glob.glob(os.path.join(predict_dir, "*.obj"))
    parallel_show_predict(predict_model_list, predict_dir, n_workers=8)

    # 显示结果
    file_list = [os.path.join(predict_dir, file_path) for file_path in os.listdir(predict_dir)
                 if os.path.isdir(os.path.join(predict_dir, file_path))]

    for i, file in enumerate(file_list):
        print("{} file path is: {}".format(i + 1, file))
        predict1_path = os.path.join(file, "predict1.obj")
        predict2_path = os.path.join(file, "predict2.obj")
        predict_pts = os.path.join(file, "predict.vtk")
        show_predict(predict1_path, predict2_path, predict_pts)

    # ----- 按键控制 --------
    # length = len(file_list)
    # i = 0
    # while True:
    #     file = file_list[i]
    #     print("\n第{}个 file path is: {}".format(i + 1, file))
    #     predict1_path = os.path.join(file, "predict1.obj")
    #     predict2_path = os.path.join(file, "predict2.obj")
    #     predict_pts = os.path.join(file, "predict.vtk")
    #     show_predict(predict1_path, predict2_path, predict_pts)
    #
    #     print("*****A(a):上一张; D(d):下一张; Q(q):退出; 按完键后回车表示确定！！！")
    #     line = sys.stdin.readline()
    #     if line == "a\n" or line == "A\n":
    #         if i > 0:
    #             i -= 1
    #         else:
    #             print("已经到最前面一张了，请按D(d)到下一张，按Q(q)退出")
    #             i = 0
    #     if line == "d\n" or line == "D\n":
    #         if i < length - 1:
    #             i += 1
    #         else:
    #             print("已经到最后面一张了，请按A(a)到下一张，按Q(q)退出")
    #             i = length - 1
    #     if line == "q\n" or line == "Q\n":
    #         break




