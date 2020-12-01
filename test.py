from options.test_options import TestOptions
from data import DataLoader
from models import create_model
import torch

from models import networks


def run_test():
    print('Running Test')
    opt = TestOptions().parse()
    dataset = DataLoader(opt)
    model = create_model(opt)

    print("data number: {} \n ".format(len(dataset)))
    for i, data in enumerate(dataset):
        print("Predict {} ".format(i+1))
        model.set_input(data)
        pred_class = model.test()
        print("Predict result :{}".format(pred_class))

    print("-----------------\npredict done!")


if __name__ == '__main__':
    run_test()
