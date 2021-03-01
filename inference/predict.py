import torch
import time
import numpy as np
import os
from mesh_net import network
from mesh_net.mesh import Mesh
import pickle
from config import config
torch.backends.cudnn.enabled = False


def data_pad(input_arr, target_length, val=0, dim=1):
    shp = input_arr.shape
    npad = [(0, 0) for _ in range(len(shp))]
    npad[dim] = (0, target_length - shp[dim])
    return np.pad(input_arr, pad_width=npad, mode='constant', constant_values=val)


def get_mean_std(mean_std_path):
    """
    Computes Mean and Standard Deviation from Training Data 'mean_std_cache.p'
    """
    if not os.path.isfile(mean_std_path):
        mean = 0
        std = 1
        ninput_channels = 5
        print("can't find the mean_std_cache.p!")
    else:
        with open(mean_std_path, 'rb') as f:
            transform_dict = pickle.load(f)
            print('loaded mean / std from cache')
            mean = transform_dict['mean']
            std = transform_dict['std']
            ninput_channels = transform_dict['ninput_channels']
    return mean, std, ninput_channels


def get_n_class(classes_file):
    classes = np.loadtxt(classes_file)
    offset = classes[0]
    classes = classes - offset
    # return classes
    return len(classes)


def export_segmentation(pred_seg, mesh):
    for meshi, mesh_ in enumerate(mesh):
        mesh_.export_segments(pred_seg[meshi, :])


def process_data(model_path, mean, std, opt):
    mesh = Mesh(file=model_path, export_folder=opt.export_folder, hold_history=True, phase="test")
    meta = dict()
    meta["filename"] = mesh.filename
    meta['mesh'] = mesh

    # get edge features
    edge_features = mesh.extract_features()
    if edge_features.shape[1] <= opt.ninput_edges:
        edge_features = data_pad(edge_features, opt.ninput_edges)
        meta['edge_features'] = (edge_features - mean) / std
    else:
        meta["edge_features"] = np.array([])

    return meta


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


def run_test():
    print('Running Test')
    # *** 1. config
    config.nclasses = get_n_class(config.class_file)
    mean, std, config.input_nc = get_mean_std(config.mean_std_file)
    # *** 2. model
    # define network
    # net = network.define_classifier(config)
    print("****")
    net = network.define_classifier(config)
    print("----")
    if isinstance(net, torch.nn.DataParallel):
        # print("net*************")
        net = net.module
    device = torch.device('cuda:{}'.format(config.gpu_ids[0])) if config.gpu_ids else torch.device('cpu')
    # load model weight
    print('loading the model from %s' % config.model_path)
    state_dict = torch.load(config.model_path, map_location=str(device))
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    net.load_state_dict(state_dict)
    net = net.eval()
    # torch.save(net, "./model_weight/predict.pth")  # 保存网络和权重
    # script_model = torch.jit.script(net)
    # print(script_model.code)

    # *** 3. data
    test_obj_lists = []
    for item in os.walk(config.test_dir):
        for file in item[2]:
            file_path = os.path.join(item[0], file)
            if os.path.splitext(file_path)[1] == ".obj":
                test_obj_lists.append(file_path)
    print("data number: {} \n ".format(len(test_obj_lists)))

    # *** 4. loop predict
    for i, model_file in enumerate(test_obj_lists):
        # base_name, ext = os.path.splitext(os.path.basename(model_file))
        # model_path_file = os.path.join(config.export_folder, base_name + "_0" + ext)
        # if os.path.isfile(model_path_file):
        #     predict_vs, predict_faces, predict_edges = parse_obje(model_path_file)
        #     if len(predict_edges) != 0:
        #         continue
        data = process_data(model_file, mean, std, config)
        print("Predict {}th file: {}".format((i + 1), data["filename"]))
        try:
            start = time.time()
            if data["edge_features"].shape[1] == 0:
                raise ValueError("input edges must greater than feature shape,"
                                 " please check your model")

            features = torch.from_numpy(data['edge_features']).float().unsqueeze(0)
            features = features.to(device).requires_grad_(False)
            mesh = [data['mesh']]

            # predict
            out = net(features, mesh)
            pred_class = out.data.max(1)[1].cpu()
            # np.savetxt("/home/heygears/work/github/MeshCNN/001", pred_class.numpy())
            # 导出结果
            export_segmentation(pred_class, mesh)

            # # compare result
            # ori_class = np.loadtxt("/home/heygears/work/github/MeshCNN/001")
            # print("***result is ture: ***", (ori_class == pred_class.numpy()).all())

            end = time.time()
            run_time = end - start
            # print("Predict result :{}, run time is {}ms".format(pred_class, run_time*1000))
        except Exception as e:
            print("error{}", repr(e))
            with open('error_model.txt', mode='a') as filename:
                filename.write(repr(e))
                filename.write(" ")
                filename.write(str(data["filename"]))
                filename.write('\n')  # 换行

    print("-----------------\npredict done!")


if __name__ == '__main__':

    run_test()
