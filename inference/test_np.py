import numpy as np
from scipy import spatial


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


def split_edges(edges, vs, threshold=30):

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
        if len(circle) < threshold:
            continue
        else:
            filter_edges.append(circle)
            filter_pts.append(circle_pts[idx])

    # ---- 3. 合并 ----
    # 长度为1直接返回, 如果长度大于1，说明有2个或者以上的闭环，这时可以先将各闭环对应的边找出，上一步骤保存即可
    # 然后利用KDTree找出两个集合中最近的两条边，断开重组（重组时根据x,y值避免缠绕）即删除两条边，构造两条虚拟边
    n_circle = len(filter_edges)
    if 1 == n_circle:
        pts_ids = []
        for circle in filter_pts:
            # 过滤短的
            if len(circle) > threshold:
                # print("{}".format(len(circle)))
                circle = drop_cycle(circle, threshold)  # 去闭环
                # print("after drop cycle {}".format(len(circle)))
                pts_ids.append(circle)

        return pts_ids
    else:
        last_edges = filter_edges[0]
        vs_edges = vs[last_edges].reshape(len(last_edges), -1)
        tree = spatial.KDTree(vs_edges)
        for i in range(1, n_circle):
            cur_edges = filter_edges[i]
            min_dist = np.inf
            min_index = -1
            cur_edge = cur_edges[0]
            cur_e_idx = 0
            for e_idx, e in enumerate(cur_edges):
                vs_e = np.append(vs[e[0]], vs[e[1]])
                dist, dist_idx = tree.query(vs_e)
                if dist < min_dist:
                    min_dist = dist
                    min_index = dist_idx
                    cur_edge = e
                    cur_e_idx = e_idx

            # 迭代
            # 上一个闭环中最近的边
            last_edge = last_edges[min_index]
            last_edge_y1 = vs[last_edge[0]][1]
            last_edge_y2 = vs[last_edge[1]][1]
            last_lower_y_idx = 0 if last_edge_y1 < last_edge_y2 else 1
            # 本次闭环中最近的边
            cur_edge_y1 = vs[cur_edge[0]][1]
            cur_edge_y2 = vs[cur_edge[1]][1]
            cur_lower_y_idx = 0 if cur_edge_y1 < cur_edge_y2 else 1
            # 根据y值重新组合两条边
            edge_1 = [[last_edge[last_lower_y_idx], cur_edge[cur_lower_y_idx]]]
            edge_2 = [[last_edge[1-last_lower_y_idx], cur_edge[1-cur_lower_y_idx]]]
            # 重新生成last_edges
            last_edges = last_edges[:min_index] + last_edges[min_index+1:] + cur_edges[:cur_e_idx] + cur_edges[cur_e_idx+1:]
            last_edges = last_edges + edge_1 + edge_2
            if i + 1 < n_circle:  # 小于才重新构建tree
                vs_edges = vs[last_edges].reshape(len(last_edges), -1)
                tree = spatial.KDTree(vs_edges)

        circle_pts = [[]]
        count = 0
        while len(last_edges) > 0:
            if len(circle_pts[count]) == 0:
                circle_pts[count] = list(last_edges[0])
                last_edges = np.delete(last_edges, 0, axis=0)
            else:
                last_id = circle_pts[count][-1]
                idx = np.where(last_edges == last_id)[0]
                # 没有找到边
                if len(idx) == 0:
                    circle_pts.append([])
                    count += 1
                else:
                    edge = last_edges[idx[0]]
                    next_id = edge[0] if edge[0] != last_id else edge[1]
                    circle_pts[count].append(next_id)
                    last_edges = np.delete(last_edges, idx[0], axis=0)
        pts_ids = []
        for circle in circle_pts:
            # 过滤短的
            if len(circle) > threshold:
                # print("{}".format(len(circle)))
                circle = drop_cycle(circle, threshold)  # 去闭环
                # print("after drop cycle {}".format(len(circle)))
                pts_ids.append(circle)
        return pts_ids


def show_obj_pts_points(obj_path, pts, max_points):
    import vedo
    stl_model = vedo.load(obj_path).c(("magenta"))
    points = vedo.Points(pts.reshape(-1, 3)).pointSize(10).c(("green"))
    p = vedo.Points(max_points.reshape(-1, 3)).pointSize(15).c(("yellow"))
    vedo.show(stl_model, points, p)


if __name__ == "__main__":

    # edges = np.loadtxt("./edges.txt", dtype=int)
    # vs = np.loadtxt("./vs.txt")
    # pts_ids = np.loadtxt("./pts_ids.txt", dtype=int)  # split_edges(edges, vs)
    # feture_vs = vs[np.array(pts_ids)]
    # mean = np.mean(feture_vs, axis=0)
    # mean_z = mean[2]
    # std = np.var(feture_vs, axis=0)
    # max_diff = 0
    # print(mean, std)
    # for i in range(1, len(pts_ids) - 1):
    #     last_pt = vs[pts_ids[i - 1]]
    #     cur_pt = vs[pts_ids[i]]
    #     last_pt_z = last_pt[2]
    #     cur_pt_z = cur_pt[2]
    #     if last_pt_z < mean_z or cur_pt_z < mean_z:
    #         continue
    #
    #     z_diff = cur_pt_z - last_pt_z
    #     if z_diff > max_diff:
    #         max_diff = z_diff
    #         print("z diff: ", z_diff, last_pt, cur_pt)
    # print("max_diff: ", max_diff)

    pts_path = "/data/Test_5044/AI_pts/BSGB2_VS_SET_VSc2_Subsetup8_Mandibular.pts"
    obj_path = "/data/Test_5044/down_obj/BSGB2_VS_SET_VSc2_Subsetup8_Mandibular.obj"
    pts_vs = get_gum_line_pts(pts_path)
    mean = np.mean(pts_vs, axis=0)
    mean_z = mean[2]
    std = np.var(pts_vs, axis=0)
    print(mean, std)
    max_diff = 0
    max_points = []
    for i in range(1, len(pts_vs) - 1):
        last_pt = pts_vs[i - 1]
        cur_pt = pts_vs[i]
        last_pt_z = last_pt[2]
        cur_pt_z = cur_pt[2]
        if last_pt_z < mean_z or cur_pt_z < mean_z:
            continue

        z_diff = cur_pt_z - last_pt_z
        if z_diff > max_diff:
            max_diff = z_diff
            print("z diff: ", z_diff, last_pt, cur_pt)
            if max_diff > 1.5:
                max_points.append(last_pt)
                max_points.append(cur_pt)
    print("max_diff: ", max_diff)
    max_points = np.asarray(max_points)
    show_obj_pts_points(obj_path, pts_vs, max_points)