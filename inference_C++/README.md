## 模型预测

### 1.文件说明
- class_features_info     存放类别文件以及特征均值和方差文件
    - classes.txt:        类别文件
    - mean_std_cache.p    训练生成的特征均值及方差
    
- data_deal               数据预处理
    - data_preprocess.py  原始stl模型转为obj格式并下采样
    - hgapi               用于协助数据处理的工具
    
- mesh_net                AI网络结构及模型处理
    - mesh.py             牙模类
    - mesh_process.py     牙模处理脚本
    - network.py          AI网络结构
    
- model_weight            AI模型权重存放路径
    - 200_net.pth         AI模型权重
    
- results                 预测结果存放路径

- test_models             待预测牙模存放路径

- config.py               AI模型相关配置

- predict.py              预测脚本

- show_result.py          查看预测结果脚本

### 2. 预测流程
- step1

    利用data_preprocess.py处理stl模型，将其转换为obj格式，并下采样到指定路径
    
    修改文件第7行中`hgapi`所在路径及以下内容
    ```python
    stl_dir = "./stl"            # stl模型存放路径
    obj_save_dir = "./obj"       # obj模型存放路径
    down_obj_save_dir = "./down_obj"      # 下采样后模型存放路径
    target_faces = 5000          # 下采样面片的个数
    ```
    然后运行
    ```bash
    $ blender --background --python data_preprocess.py
    ```
    
- step2

    第一步中下采样后的模型放入`test_models`目录下
     
- step3

    运行预测脚本
    ```bash
    $ python predict.py
    ```
  
- step4

    检测结果自动保存在`./results`目录下
    
- step5

    查看检测结果
    ```bash
    $ python show_result.py
    ```
  
### 3. 相关依赖
  - blender      2.90.1
  - python       >= 3.6.5
     - vedo        2020.4.1
     - torch        >= 1.2.0
    