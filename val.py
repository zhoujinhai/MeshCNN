from options.val_options import ValOptions
from data import DataLoader
from models import create_model
from util.writer import Writer


def run_val(epoch=-1):
    print('Running Val')
    opt = ValOptions().parse()
    # opt.gpu_ids = []  # use cpu [], gpu: [0]
    dataset = DataLoader(opt)
    model = create_model(opt)
    writer = Writer(opt)
    # val
    writer.reset_counter()

    for i, data in enumerate(dataset):
        model.set_input(data)
        try:
            ncorrect, nexamples = model.val()
            writer.update_counter(ncorrect, nexamples)
        except Exception as e:
            print(e)
            print(data["filename"])
            with open('error_model.txt', mode='a') as filename:
                filename.write(str(data["filename"][0]))
                filename.write('\n')  # 换行
    writer.print_acc(epoch, writer.acc)
    return writer.acc


if __name__ == '__main__':
    run_val()
