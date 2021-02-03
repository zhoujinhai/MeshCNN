class config(object):
    # ---- network ----
    ncf = [32, 64, 128, 256]       # 卷积层通道数
    resblocks = 3                  # 卷积层重复次数
    pool_res = [7000, 5000, 3500]  # 池化层
    fc_n = 100                     # 全连接层节点数
    ninput_edges = 7500            # 输入边数
    init_type = 'normal'           # 初始化方式 'normal', 'xavier', 'kaiming', 'orthogonal'
    init_gain = 0.02               # 增益率
    input_nc = 7                   # 输入特征维度
    nclasses = 2                   # 类别数

    # ---- device ----
    gpu_ids = [0]                   # gpu:[0]  cpu:[]

    # ---- 类别和平均值文件 ----
    class_file = "./class_features_info/classes.txt"
    mean_std_file = "./class_features_info/mean_std_cache.p"

    # ---- file path ----
    # # 模型路径
    model_path = "./model_weight/200_net.pth"
    # # 测试集存放路径
    test_dir = "./test_models"
    # # 模型结果保存路径
    export_folder = './results'

