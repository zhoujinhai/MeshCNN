# Data FLower

## dataset = DataLoader(opt)

## dataset = CreateDataset(opt)

### ClassificationData(opt)

### SegmentationData(opt)

## __next__()

## __iter__()

## __collate_fn()

## __getitem__()

## Mesh.__init__()

### fill_mesh()

给mesh赋值 包括vs,edges,gemm_edges,features等

### get_mesh_path()

获取对应obj文件对应的npz文件

- 存在obj对应的npz文件,解析
- 不存在

## from_scratch()

### fill_from_file()

解析obj文件获取vs，faces

### remove_non_manifolds()

移除非正常的faces

### augmentation()

数据增强

### build_gemm()

获取相邻边

### extract_feature()

提取特征

*XMind - Trial Version*