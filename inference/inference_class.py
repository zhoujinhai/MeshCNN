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
from mesh_net import deal_result

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

    def process_data(self, model_path, opt, is_obj=False):
        mesh = Mesh(file=model_path, export_folder=opt.export_folder, hold_history=True, phase="test", target_edges=opt.ninput_edges, is_obj=is_obj)
        meta = dict()
        meta["filename"] = mesh.filename
        meta['mesh'] = mesh
        meta["vs"], meta["faces"], meta["edges"] = mesh.get_vs_faces_edges()
        print("mesh info: ", len(meta["vs"]), len(meta["faces"]), len(meta["edges"]))

        # get edge features
        edge_features = mesh.extract_features()

        if edge_features.shape[1] <= opt.ninput_edges:
            edge_features = data_pad(edge_features, opt.ninput_edges)
        meta['edge_features'] = (edge_features - self.mean) / self.std

        self.meta = meta

    def inference_obj(self, obj_data):
        with torch.no_grad():
            try:
                # *** 3. process data
                start = time.time()
                self.process_data(obj_data, config, is_obj=True)

                # *** 4. predict
                if len(self.meta["edges"]) > config.ninput_edges:
                    raise ValueError("input faces must less than or equal to {},"
                                     " please check your down sample obj model".format(config.ninput_edges))

                features = torch.from_numpy(self.meta['edge_features']).float().unsqueeze(0)
                # features = features.half()  # TODO half()
                # print("features after half: ", features.dtype)
                features = features.to(self.device).requires_grad_(False)
                mesh = [self.meta['mesh']]

                # predict
                out = self.net(features, mesh)

                pred_class = out.data.max(1)[1].cpu()

                end = time.time()
                run_time = end - start
                print("Predict result :{}, \nrun time is: {}ms".format("success!", run_time * 1000))

                return pred_class.tolist()

            except Exception as e:
                print("predict error: ", e)
                return []

    def inference(self, model_path):
        with torch.no_grad():
            # frame = inspect.currentframe()
            # gpu_tracker = MemTracker(frame)
            # gpu_tracker.track()

            try:
                # *** 3. process data
                start = time.time()
                self.process_data(model_path, config)
                # gpu_tracker.track()

                # *** 4. predict
                if len(self.meta["edges"]) > config.ninput_edges:
                    raise ValueError("input faces must less than or equal to {},"
                                     " please check your down sample obj model".format(config.ninput_edges))

                features = torch.from_numpy(self.meta['edge_features']).float().unsqueeze(0)
                # features = features.half()  # TODO half()
                # print("features after half: ", features.dtype)
                features = features.to(self.device).requires_grad_(False)
                mesh = [self.meta['mesh']]

                # predict
                out = self.net(features, mesh)

                pred_class = out.data.max(1)[1].cpu()
                # 导出结果
                # export_segmentation(pred_class, mesh)

                # 获取pts
                # predict_labels = pred_class[0].numpy()
                # print("predict_labels: ", predict_labels, len(predict_labels))
                # print(self.meta["edge_cnt"])
                # np.savetxt("./vs.txt", self.meta["vs"], fmt="%6f")
                # np.savetxt("./faces.txt", self.meta["faces"], fmt="%d")
                # np.savetxt("./edges.txt", self.meta["edges"], fmt="%d")
                # np.savetxt(r"D:\Debug_dir\test\label.txt", predict_labels, fmt="%d")
                # predict_gum_pts = deal_result.get_predict_pts(self.meta["vs"], self.meta["faces"],
                #                                               self.meta["edges"], predict_labels)

                end = time.time()
                run_time = end - start
                print("Predict result :{}, \nrun time is: {}ms".format("success!", run_time * 1000))
                # pts_path = os.path.splitext(os.path.basename(model_path))[0] + ".pts"
                # np.savetxt(os.path.join("./AI_pts", pts_path), predict_gum_pts)
                # np.savetxt(pts_path, predict_gum_pts)

                # 返回边标签
                return pred_class.tolist()
                # 返回牙龈线点
                # return predict_gum_pts.tolist()

            except Exception as e:
                print("predict {} exited with error {}".format(self.meta["filename"], repr(e)))
                # with open('./ai_error_model.txt', mode='a') as filename:
                #     filename.write(repr(e))
                #     filename.write(" ")
                #     filename.write(str(model_path))
                #     filename.write('\n')  # 换行
                return []


if __name__ == '__main__':
    # test_file = "/data/Test_5044/down_obj/C2SFP_VS_SET_VSc1_Subsetup_Retainer_Mandibular.obj"
    test_file = r"D:\Debug_dir\test\test.obj"
    test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), test_file)
    print("Test file: ", test_file)
    inference_class = InferenceClass()
    for i in range(1):
        print("------predict {} times-----".format(i + 1))
        predict_class = inference_class.inference(test_file)
        result1 = np.array(predict_class)
        np.savetxt('predict_class.pts', predict_class)

    # import glob
    # test_files = glob.glob(os.path.join("/data/Test_5044/down_obj/", "*.obj"))
    # inference_class = InferenceClass()
    #
    # for i, test_file in enumerate(test_files):
    #      print("{}  {}".format(i, test_file))
    #      predict_class = inference_class.inference(test_file)


