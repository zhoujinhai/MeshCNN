from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from util.writer import Writer


def run_test(epoch=-1):
    print('Running Test')
    opt = TestOptions().parse()
    dataset = DataLoader(opt)
    model = create_model(opt)
    writer = Writer(opt)
    # test
    writer.reset_counter()
    pred_classes = []
    for i, data in enumerate(dataset):
        model.set_input(data)
        pred_class = model.test()
        print("pred_class", pred_class, len(pred_class))
        pred_classes.append(pred_class)
    print("predict done!")
    return pred_classes


if __name__ == '__main__':
    run_test()
