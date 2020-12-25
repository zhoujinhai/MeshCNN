### MeshCNN测试流程

#### 1.数据预处理

***注意：先修改stl路径， obj保存路径 和 测试集路径***

```
stl_model_dir = " "   # stl文件保存路径
obj_save_dir = ""     # stl转obj后保存路径
test_dir = ""         # 测试集存放路径 
```

然后运行

```python
blender --background --python data_preprocess.py
```

#### 2. 利用AI模型进行预测

注意根据实际情况修改以下内容

- --name:   AI模型保存路径
- ncf: 网络结构
- pool_res: 池化层
- which_epoch: AI模型序号

```shell
python test.py \
--dataroot datasets/tooth_seg \
--name tooth_seg_20201215 \
--arch meshunet \
--dataset_mode segmentation \
--ncf 16 32 64 128 256 \
--ninput_edges 7500 \
--pool_res 7000 6000 4500 3000 \
--resblocks 3 \
--batch_size 1 \
--export_folder meshes \
--which_epoch 100
```

运行结束后会在--name目录下生成一个meshs文件夹，里面存放预测后结果

#### 3.查看预测结果

修改`show_data.py`中predict_dir路径为第二步中生成的meshs所在路径，如: predict_dir = ".../meshs"

然后运行

```python
python show_data.py
```

