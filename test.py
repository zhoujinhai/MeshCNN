from options.test_options import TestOptions
from data import DataLoader
from models import create_model
import torch

from models import networks
import time


def run_test():
    print('Running Test')
    opt = TestOptions().parse()
    dataset = DataLoader(opt)
    model = create_model(opt)

    print("data number: {} \n ".format(len(dataset)))
    for i, data in enumerate(dataset):
        print("Predict {} ".format(i+1))
        # try:
        start = time.time()
        model.set_input(data)
        pred_class = model.test()
        end = time.time()
        run_time = end - start
        print("Predict result :{}, run time is {}ms".format(pred_class, run_time*1000))
        # except Exception as e:
        #     print(e)
        #     print(data["filename"])
        #     with open('error_model.txt', mode='a') as filename:
        #         filename.write(str(data["filename"][0]))
        #         filename.write('\n')  # 换行

    print("-----------------\npredict done!")


if __name__ == '__main__':
    run_test()
