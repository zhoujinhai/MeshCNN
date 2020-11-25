from options.val_options import ValOptions
from data import DataLoader
from models import create_model
from util.writer import Writer


def run_val(epoch=-1):
    print('Running Val')
    opt = ValOptions().parse()
    dataset = DataLoader(opt)
    model = create_model(opt)
    writer = Writer(opt)
    # test
    writer.reset_counter()

    for i, data in enumerate(dataset):
        model.set_input(data)
        ncorrect, nexamples = model.val()
        writer.update_counter(ncorrect, nexamples)
    writer.print_acc(epoch, writer.acc)
    return writer.acc


if __name__ == '__main__':
    run_val()
