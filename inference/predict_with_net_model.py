import torch
import time
import numpy as np
import os
from mesh_net.mesh import Mesh
import pickle
from config import config

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
            # print('loaded mean / std from cache')
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


def run_test():
    print('Running Test')
    # *** 1. config
    config.nclasses = get_n_class(config.class_file)
    mean, std, config.input_nc = get_mean_std(config.mean_std_file)

    # *** 2. load model
    device = torch.device('cuda:{}'.format(config.gpu_ids[0])) if config.gpu_ids else torch.device('cpu')
    # load model weight
    net = torch.load("./model_weight/predict.pth", map_location=str(device))
    # print(net)
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
            # 导出结果
            export_segmentation(pred_class.cpu(), mesh)

            end = time.time()
            run_time = end - start
            print("Predict result :{}, run time is {}ms".format(pred_class, run_time*1000))
        except Exception as e:
            print("error{}", repr(e))
            with open('error_model.txt', mode='a') as filename:
                filename.write(repr(e))
                filename.write(" ")
                filename.write(str(data["filename"]))
                filename.write('\n')  # 换行

    print("-----------------\npredict done!")


def inference(model_path):
    # *** 1. load config
    config.nclasses = get_n_class(config.class_file)
    mean, std, config.input_nc = get_mean_std(config.mean_std_file)
    # print("mean: {}, std: {}, input_nc: {}".format(mean, std, config.input_nc))
    print("load config successed!")

    # *** 2. load model
    device = torch.device('cuda:{}'.format(config.gpu_ids[0])) if config.gpu_ids else torch.device('cpu')
    net = torch.load("./model_weight/predict.pth", map_location=str(device))
    net = net.eval()
    print("load model successed!")

    # *** 3. process model
    data = process_data(model_path, mean, std, config)

    # *** 4. predict
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
        pred_class = out.data.cpu()  # .max(1)[1].cpu()
        # 导出结果
        # export_segmentation(pred_class.cpu(), mesh)

        end = time.time()
        run_time = end - start
        print("Predict result :{}, \nrun time is {}ms".format(pred_class, run_time * 1000))
        return pred_class

    except Exception as e:
        print("predict {} exited with error {}".format(data["filename"], repr(e)))


if __name__ == '__main__':

    model_file = "/home/heygears/work/github/MeshCNN/inference/test_models/1AQ9W_VS_SET_VSc1_Subsetup6_Maxillar.obj"
    pred_res = inference(model_file)
    class_res = pred_res.max(1)[1]
    print("class: {}".format(class_res))

    # run_test()
