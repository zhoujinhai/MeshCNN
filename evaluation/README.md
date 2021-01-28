### 模型评估

#### 1、evaluate_testsets.py
评估测试集准确度。

- 使用时先修改如下内容：
```python
pts_dir = "/home/heygears/work/Tooth_data_prepare/tooth/three_batch_data/pts" # 原始牙龈线文件路径
predict_model_dir = "/home/heygears/work/github/MeshCNN/inference/results" # 模型预测后存放路径
threshold = 2  # 距离阈值，用于筛选
```
- 然后运行
```bash
$ python evaluate_testsets.py
```

#### 2、show_pts.py
查看原始牙龈线是否正确, 如果不正确需要根据前端streamFlow进行调整，
然后下载对应的stl和pts文件，再对原先的文件进行替换<font color="red">替换需stl和pts同时替换</font>

- 先修改文件中路径
```python
# pts文件存放路径
pts_path = "/home/heygears/work/Tooth_data_prepare/tooth/three_batch_data/pts"         
# obj存放路径，如果是stl文件，则再下面一行中相应的修改"*.obj"为"*.stl"
obj_path = "/home/heygears/work/Tooth_data_prepare/tooth/three_batch_data/file/train"
# 中间文件存放路径，使用完可以删除
vtk_path = "/home/heygears/work/Tooth_data_prepare/tooth/three_batch_data/vtk"
pts_list = glob.glob(os.path.join(pts_path, "*.pts"))
obj_list = glob.glob(os.path.join(obj_path, "*.obj"))
```
- 运行
```bash
$ python show_pts.py
```
