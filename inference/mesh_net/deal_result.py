#!/usr/bin/env python
# coding: utf-8
""""
Usage: python show_data.py
"""

# In[1]:

import numpy as np
from scipy import spatial
import math
import sys
import os


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
    circle_edges = [[]]
    count = 0
    while len(edges) > 0:
        if len(circle_pts[count]) == 0:
            circle_pts[count] = list(edges[0])
            circle_edges[count].append(list(edges[0]))   # 记录对应的边
            edges = np.delete(edges, 0, axis=0)
        else:
            last_id = circle_pts[count][-1]
            idx = np.where(edges == last_id)[0]
            # 没有找到边
            if len(idx) == 0:
                circle_pts.append([])
                circle_edges.append([])
                count += 1
            else:
                edge = edges[idx[0]]
                next_id = edge[0] if edge[0] != last_id else edge[1]
                circle_pts[count].append(next_id)
                circle_edges[count].append(list(edge))
                edges = np.delete(edges, idx[0], axis=0)

    pts_ids = []
    for circle in circle_pts:
        # 过滤短的
        if len(circle) > threshold:
            # print("{}".format(len(circle)))
            circle = drop_cycle(circle, threshold)   # 去闭环
            # print("after drop cycle {}".format(len(circle)))
            pts_ids.append(circle)

    return pts_ids


def get_pts_from_edges_vs(edges, vs, threshold=30):
    # ---- split_edges ----
    circle_pts = [[]]
    circle_edges = [[]]
    count = 0
    while len(edges) > 0:
        if len(circle_pts[count]) == 0:
            circle_pts[count] = list(edges[0])
            circle_edges[count].append(edges[0])   # 记录对应的边
            edges = np.delete(edges, 0, axis=0)
        else:
            last_id = circle_pts[count][-1]
            idx = np.where(edges == last_id)[0]
            # 没有找到边
            if len(idx) == 0:
                circle_pts.append([])
                circle_edges.append([])
                count += 1
            else:
                edge = edges[idx[0]]
                next_id = edge[0] if edge[0] != last_id else edge[1]
                circle_pts[count].append(next_id)
                circle_edges[count].append(edge)
                edges = np.delete(edges, idx[0], axis=0)

    # ---- 2.过滤掉长度不符合要求的 （噪声点）----
    filter_edges = []
    filter_pts = []
    for idx, circle in enumerate(circle_edges):
        # print(len(circle))
        if len(circle) < threshold:
            continue
        else:
            filter_edges.append(circle)
            circle_pt = drop_cycle(circle_pts[idx], threshold)
            filter_pts.append(circle_pt)

    # # save pts
    # for idx, pts_id in enumerate(filter_pts):
    #     save_dir = "./test_circle"
    #     pts = vs[pts_id]
    #     np.savetxt(os.path.join(save_dir, "predict_class" + str(idx + 1) + ".pts"), pts)
    #     with open(os.path.join(save_dir, "predict_class" + str(idx + 1) + ".pts"), 'r+') as f:
    #         content = f.read()
    #         f.seek(0, 0)
    #         f.write('BEGIN\n' + content)
    #     with open(os.path.join(save_dir, "predict_class" + str(idx + 1) + ".pts"), 'a') as f:
    #         f.write('END\n')

    return filter_pts

    # # ---- 3. 合并 ----
    # # 长度为1直接返回, 如果长度大于1，说明有2个或者以上的闭环（3个或以上需要按x顺序合并），这时可以先将各闭环对应的边找出，上一步骤保存即可
    # # 然后利用KDTree找出两个集合中最近的两条边，断开重组（重组时根据x,y值避免缠绕）即删除两条边，构造两条虚拟边
    # n_circle = len(filter_edges)
    # if 0 == n_circle:
    #     return []
    # elif 1 == n_circle:
    #     pts_ids = []
    #     for circle in filter_pts:
    #         circle = drop_cycle(circle, threshold)  # 去闭环
    #         pts_ids.append(circle)
    #     return pts_ids
    # else:
    #     # TODO 3个以上根据x的范围进行排序，可过滤掉包含在大圈里面的部分 (这样处理有些有问题 如：APDXA_VS_SET_VSc2_Subsetup4_Maxillar)
    #     # 找最近的边,进行破圈
    #     last_edges = filter_edges[0]
    #     vs_edges = vs[last_edges].reshape(len(last_edges), -1)
    #     tree = spatial.KDTree(vs_edges)
    #     for i in range(1, n_circle):
    #         cur_edges = filter_edges[i]
    #         min_dist = np.inf
    #         min_index = -1
    #         cur_edge = cur_edges[0]
    #         cur_e_idx = 0
    #         for e_idx, e in enumerate(cur_edges):
    #             vs_e = np.append(vs[e[0]], vs[e[1]])
    #             dist, dist_idx = tree.query(vs_e)
    #             if dist < min_dist:
    #                 min_dist = dist
    #                 min_index = dist_idx
    #                 cur_edge = e
    #                 cur_e_idx = e_idx
    #
    #         # 迭代
    #         # 上一个闭环中最近的边
    #         last_edge = last_edges[min_index]
    #         last_edge_y1 = vs[last_edge[0]][1]
    #         last_edge_y2 = vs[last_edge[1]][1]
    #         last_lower_y_idx = 0 if last_edge_y1 < last_edge_y2 else 1
    #         # 本次闭环中最近的边
    #         cur_edge_y1 = vs[cur_edge[0]][1]
    #         cur_edge_y2 = vs[cur_edge[1]][1]
    #         cur_lower_y_idx = 0 if cur_edge_y1 < cur_edge_y2 else 1
    #         # 根据y值重新组合两条边
    #         edge_1 = [[last_edge[last_lower_y_idx], cur_edge[cur_lower_y_idx]]]
    #         edge_2 = [[last_edge[1-last_lower_y_idx], cur_edge[1-cur_lower_y_idx]]]
    #         # 重新生成last_edges
    #         last_edges = last_edges[:min_index] + last_edges[min_index+1:] + cur_edges[:cur_e_idx] + cur_edges[cur_e_idx+1:]
    #         last_edges = last_edges + edge_1 + edge_2
    #         if i + 1 < n_circle:  # 小于才重新构建tree
    #             vs_edges = vs[last_edges].reshape(len(last_edges), -1)
    #             tree = spatial.KDTree(vs_edges)
    #
    #     # 按边将点拼接成一个闭环
    #     circle_pts = [[]]
    #     count = 0
    #     while len(last_edges) > 0:
    #         if len(circle_pts[count]) == 0:
    #             circle_pts[count] = list(last_edges[0])
    #             last_edges = np.delete(last_edges, 0, axis=0)
    #         else:
    #             last_id = circle_pts[count][-1]
    #             idx = np.where(last_edges == last_id)[0]
    #             # 没有找到边
    #             if len(idx) == 0:
    #                 circle_pts.append([])
    #                 count += 1
    #             else:
    #                 edge = last_edges[idx[0]]
    #                 next_id = edge[0] if edge[0] != last_id else edge[1]
    #                 circle_pts[count].append(next_id)
    #                 last_edges = np.delete(last_edges, idx[0], axis=0)
    #     pts_ids = []
    #     for circle in circle_pts:
    #         # 过滤短的
    #         if len(circle) > threshold:
    #             # print("{}".format(len(circle)))
    #             circle = drop_cycle(circle, threshold)  # 去闭环
    #             # print("after drop cycle {}".format(len(circle)))
    #             pts_ids.append(circle)
    #     return pts_ids


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
        try:
            first_id = drop_list.index(item)
            last_id = drop_list[::-1].index(item)
            if first_id + last_id <= max_length:
                length = len(drop_list)
                drop_list = drop_list[first_id:length-last_id]
        except ValueError:
            continue

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
        # TODO 对于边界边会误删
        if len(face_ids) == 2:
            if face_labels[face_ids[0][0]] != face_labels[face_ids[1][0]]:
                edge_idx.append(ei)
    test_edges = np.asarray(edges[edge_idx])
    # print("test_edges:", len(test_edges))
    pts_ids = get_pts_from_edges_vs(test_edges, vs, 10)
    # np.savetxt("./pts_ids.txt", pts_ids, fmt="%d")
    # np.savetxt("./vs.txt", vs)
    # pts_ids = get_pts_from_edges(test_edges)
    # print("pts_ids: ", pts_ids)

    res_vs = np.array([])
    face_normals, face_areas = compute_face_normals_and_areas(vs, faces)  # 计算面的法向量
    for idx, pts_id in enumerate(pts_ids):
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
            theta = math.acos(y / x) / math.pi * 180  # 不存在重合点 x可以不用判零
            if theta > 50:
                curvature = compute_pt_curvature(vs, edges, faces, face_normals, cur_pt)
                if max(curvature) > 0:
                    temp.append(cur_pt)
        temp.append(pts_id[-1])
        res_vs = np.append(res_vs, vs[temp])
        if idx != len(pts_ids) - 1:
            res_vs = np.append(res_vs, np.array([0, 0, 0]))

    return res_vs


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


def get_predict_pts(predict_vs, predict_faces, predict_edges, predict_labels):
    # ## 标记预测的面
    predict_face_labels = label_face_by_edge(predict_faces, predict_edges, predict_labels)
    # print("predict_face_labels:", len(predict_face_labels))
    # ## 方案二 通过面的标签来判断
    predict_gum_pts = label_pts_by_edges_and_faces(predict_vs, predict_edges, predict_faces, predict_face_labels)
    return predict_gum_pts.reshape(-1, 3)



