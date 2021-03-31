#import inspect
#from gpu_mem_track import MemTracker
import time
import numpy as np
import torch
import pickle
from config import config
import os
import sys
from mesh_net.mesh import Mesh

import warnings
warnings.filterwarnings("ignore", category=Warning)


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


class InferenceClass(object):
    def __init__(self):
        # ---- set file path ----
        current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
        config.class_file = os.path.join(current_path, config.class_file)
        config.mean_std_file = os.path.join(current_path, config.mean_std_file)
        config.mesh_net = os.path.join(current_path, config.mesh_net)

        # *** 1. load config
        config.nclasses = get_n_class(config.class_file)
        self.mean, self.std, config.input_nc = get_mean_std(config.mean_std_file)
        print("load config successed!")

        # *** 2. load model
        # frame = inspect.currentframe()
        # gpu_tracker = MemTracker(frame)
        # gpu_tracker.track()
        self.device = torch.device('cuda:{}'.format(config.gpu_ids[0])) \
            if config.gpu_ids and torch.cuda.is_available() else torch.device('cpu')
        self.net = torch.load(config.mesh_net, map_location=str(self.device))
        # self.net = self.net.half()  # TODO half()
        self.net = self.net.eval()
        # gpu_tracker.track()

        print("load model successed!")

        self.meta = dict()

    def process_data(self, model_path, opt):
        mesh = Mesh(file=model_path, export_folder=opt.export_folder, hold_history=True, phase="test")
        meta = dict()
        meta["filename"] = mesh.filename
        meta['mesh'] = mesh

        # get edge features
        edge_features = mesh.extract_features()
        if edge_features.shape[1] <= opt.ninput_edges:
            edge_features = data_pad(edge_features, opt.ninput_edges)
            meta['edge_features'] = (edge_features - self.mean) / self.std
        else:
            meta["edge_features"] = np.array([])

        self.meta = meta

    def inference(self, model_path):
        with torch.no_grad():
            # frame = inspect.currentframe()
            # gpu_tracker = MemTracker(frame)
            # gpu_tracker.track()
            # *** 3. process data
            self.process_data(model_path, config)
            # gpu_tracker.track()

            # *** 4. predict
            try:
                start = time.time()
                if self.meta["edge_features"].shape[1] == 0:
                    raise ValueError("input edges must greater than feature shape,"
                                     " please check your model")

                features = torch.from_numpy(self.meta['edge_features']).float().unsqueeze(0)
                # features = features.half()  # TODO half()
                # print("features after half: ", features.dtype)
                features = features.to(self.device).requires_grad_(False)
                mesh = [self.meta['mesh']]

                # predict
                out = self.net(features, mesh)

                print("out: ", out.dtype)
                pred_class = out.data.cpu()  # .max(1)[1].cpu()
                # 导出结果
                # export_segmentation(pred_class.max(1)[1], mesh)

                end = time.time()
                run_time = end - start
                print("Predict result :{}, \nrun time is {}ms".format(pred_class, run_time * 1000))

                return pred_class.tolist()

            except Exception as e:
                print("predict {} exited with error {}".format(self.meta["filename"], repr(e)))


if __name__ == '__main__':
    #test_file = "test_models/1AQ9W_VS_SET_VSc1_Subsetup6_Maxillar.obj"
    #test_file = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), test_file)

    #print("Test file: ", test_file)
    inference_class = InferenceClass()
    while True:
        test_file = input("Enter file path: ")
        test_file = test_file.strip()
        
        if test_file == "q":
            break
        if not os.path.isfile(test_file):
            print("file is not existed!")
            continue

        try:
            predict_class = inference_class.inference(test_file)
        except:
            print("Predict failed!")

