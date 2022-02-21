import os
import shutil
import sys
import glob

sys.path.append("./Release")
import hgapi

DATA_DIR = r"\\10.99.11.210\MeshCNN\MeshCNN_Train_data\down_obj"
SAVE_DIR = r"\\10.99.11.210\MeshCNN\Test_5044\MeshCNN_Train_data\file\test1"
TRAIN_DIR = r"\\10.99.11.210\MeshCNN\MeshCNN_Train_data\NewData\file\train"
VAL_DIR = r"\\10.99.11.210\MeshCNN\MeshCNN_Train_data\NewData\file\val"
val_txt = r"\\10.99.11.210\MeshCNN\MeshCNN_Train_data\NewData\file\1.txt"


if __name__ == "__main__":
    # obj_files = glob.glob(os.path.join(DATA_DIR, "*.obj"))
    # obj_handler = hgapi.OBJFileHandler()
    # n = 0
    # for idx, obj_file in enumerate(obj_files):
    #     mesh = hgapi.Mesh()
    #     obj_handler.Read(obj_file, mesh)
    #     mesh.GenerateEdgeAndAdjacency()
    #     n_edges = mesh.GetNumEdge()
    #     print(n_edges)
    #     if n_edges > 7500:
    #         print("******", idx, obj_file, n_edges)
    #         # shutil.move(obj_file, SAVE_DIR)
    #         n += 1
    # print("n: ", n)

    with open(val_txt, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            obj_path = os.path.join(TRAIN_DIR, line + ".obj")
            if os.path.isfile(obj_path):
                shutil.move(obj_path, VAL_DIR)