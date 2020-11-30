from vedo import load, show
import os


def show_predict(predict1, predict2, pts):
    """
    显示预测结果
    @predict1: 牙齿部分
    @predict2: 牙龈部分
    @pts: 牙龈线点
    return: None
    """
    a = load(predict1).c(('blue'))
    b = load(predict2).c(('magenta'))
    c = load(pts).pointSize(10).c(('green'))

    show(a, b, c)


def show_origin(origin1, origin2, pts1, pts2):
    """
    显示原始模型分割结果
    @origin1: 牙齿部分
    @origin2: 牙龈部分
    @pts1: 通过预测点投影所得牙龈线点
    @pts2: 通过边投影所得牙龈线点
    return: None
    """
    a = load(origin1).c(('blue'))
    b = load(origin2).c(('magenta'))
    c = load(pts2).pointSize(10).c(('green'))  # 边投影
    d = load(pts1).pointSize(10).c(('red'))    # 点投影

    show(a, b, c, d)


if __name__ == "__main__":

    # # ---- Test one ----
    # # predict
    # a = load('/home/heygears/work/Tooth_data_prepare/model_test/predict1.obj').c(('blue'))
    # b = load('/home/heygears/work/Tooth_data_prepare/model_test/predict2.obj').c(('magenta'))
    # c = load("/home/heygears/work/Tooth_data_prepare/model_test/predict.vtk").pointSize(10).c(('green'))
    # show(a, b, c)
    #
    # # origin
    # a = load('/home/heygears/work/Tooth_data_prepare/model_test/origin1.obj').c(('blue'))
    # b = load('/home/heygears/work/Tooth_data_prepare/model_test/origin2.obj').c(('magenta'))
    # c = load("/home/heygears/work/Tooth_data_prepare/model_test/origin2.vtk").pointSize(10).c(('green'))  # 边投影
    # d = load("/home/heygears/work/Tooth_data_prepare/model_test/origin1.vtk").pointSize(10).c(('red'))  # 点投影
    # show(a, b, c, d)

    # ---- Test Batch ----
    model_path = "/home/heygears/work/Tooth_data_prepare/model_test/predict_mesh_obj/"

    file_list = [os.path.join(model_path, file_path) for file_path in os.listdir(model_path)]

    for i, file in enumerate(file_list):
        if not os.path.isdir(file):
            continue

        predict1_path = os.path.join(file, "predict1.obj")
        predict2_path = os.path.join(file, "predict2.obj")
        predict_pts = os.path.join(file, "predict.vtk")
        show_predict(predict1_path, predict2_path, predict_pts)

        origin1_path = os.path.join(file, "origin1.obj")
        origin2_path = os.path.join(file, "origin2.obj")
        origin1_pts = os.path.join(file, "origin1.vtk")
        origin2_pts = os.path.join(file, "origin2.vtk")
        show_origin(origin1_path, origin2_path, origin1_pts, origin2_pts)


