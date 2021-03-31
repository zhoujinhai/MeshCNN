借助`aiohttp`模块将inference_class.py打包成一个web服务，用于预测。

### 1. 启动服务所需配置

| 依赖    | 版本        | 说明              |
| ------- | ----------- | ----------------- |
| python  | 3.7.4       |                   |
| torch   | 1.5.0       | python第三方库    |
| aiohttp | 3.7.4.post0 | python第三方库    |
| cuda    | 10.2        | 使用GPU才需要安装 |

- 安装python

  可借助anaconda3进行安装，具体可以参照https://www.cnblogs.com/xiaxuexiaoab/p/14544511.html第8个安装anconda3，新建一个python版本为3.7.4的虚拟环境。

- 安装torch

   进入虚拟环境，然后执行以下命令进行安装

  ```python
  pip install torch==1.5.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```

  如果安装失败，可以在<a href="\\10.99.11.210\MeshCNN\software">服务器210</a>找到对应版本的`torch-1.5.0-cp37-cp37m-*.whl`文件，然后运行

  ```python
  pip install torch-1.5.0-cp37-cp37m-*.whl
  ```

- 安装aiohttp

   ```python
  pip install aiohttp -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

- 安装cuda

  具体可以参照https://www.cnblogs.com/xiaxuexiaoab/p/14544511.html第6个安装cuda

### 2 使用说明

#### 2.1 启动服务

```bash
$ python webserver.py
```

#### 2.2 客户端测试

在`example_webclient.py`中配置好IP地址和端口号，然后运行以下命令

```bash
$ python example_webclient.py
```



