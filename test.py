from options.test_options import TestOptions
from data import DataLoader
from models import create_model
import torch
from thop import profile
from models import networks
import time
import numpy as np


def run_test():
    print('Running Test')
    opt = TestOptions().parse()
    opt.gpu_ids = []   # use cpu: [], gpu: [0]
    dataset = DataLoader(opt)
    model = create_model(opt)

    print("data number: {} \n ".format(len(dataset)))
    for i, data in enumerate(dataset):
        print("Predict {} ".format(i+1))
        try:
            start = time.time()
            print("test {}".format(data["filename"]))
            if data["edge_features"].shape[1] == 0:
                raise ValueError("input edges must greater than feature shape,"
                                 " please check your model")

            # # >>> calculation the model macs and params
            # features = torch.from_numpy(data['edge_features']).float()
            # features = features.to(model.device).requires_grad_(model.is_train)
            # mesh = data['mesh']
            # print("mesh", mesh)
            # print("mesh type: ", type(mesh))
            # traced_model = torch.jit.trace(model.net, (features, mesh, ))
            # print(traced_model.code)
            # macs, params = profile(model.net, (features, mesh, ))
            # print("macs: {}, params: {}".format(macs, params))

            model.set_input(data)
            pred_class = model.test()
            # ori_class = np.loadtxt("./001")
            # print("***result is ture: ***", (ori_class == pred_class.numpy()).all())
            end = time.time()
            run_time = end - start
            print("Predict result :{}, run time is {}ms".format(pred_class, run_time*1000))
        except Exception as e:
            print(repr(e))
            with open('error_model.txt', mode='a') as filename:
                filename.write(repr(e))
                filename.write(" ")
                filename.write(str(data["filename"][0]))
                filename.write('\n')  # 换行

    print("-----------------\npredict done!")


if __name__ == '__main__':
    run_test()
