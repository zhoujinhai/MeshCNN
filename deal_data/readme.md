### MeshCNN数据集制作
#### 目标
- 输入: stl模型及对应的牙龈线pts
- 输出: 下采样后的obj模型, 模型边对应的边标签文件eseg及邻边关系文件seseg

#### deal_file.ipynb
**功能**

- 从各文件夹中挑选出stl模型及牙龈线文件pts
- stl模型批量转换为obj格式

#### down_sample.py
**功能**

- 对obj模型进行下采样

#### deal_data_Class.ipynb
**功能**

- 根据obj模型及牙龈线pts文件生成eseg和seseg文件

#### up_label.ipynb

**功能**

- 解析预测的模型，生成分割结果
- 预测模型结果上采样回原始模型

#### show_data.py

**功能**

- 查看数据标注是否正确

#### Mesh_Simplifier.py

**功能**

- stl直接下采样生成obj牙模