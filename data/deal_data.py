import numpy as np
from models.layers.mesh_prepare import from_scratch
from models.layers.mesh_prepare import get_mesh_path
from options.base_options import BaseOptions
import os
import glob
import stl


def save_new_edges(filename):
    opt = BaseOptions()
    opt.initialize()
    opt.num_aug = 1
    mesh = from_scratch(filename, opt)
    print(dir(mesh))
    edges = np.array(mesh.edges)
    edge_file = filename.replace('.obj', '.edges')
    np.savetxt(edge_file, edges, fmt='%d')

    vs = np.array(mesh.vs)
    v_file = filename.replace('.obj', '.vs')
    np.savetxt(v_file, vs, fmt='%f')


class FakeOptions:
    def __init__(self, flip_edges=0.1):
        self.flip_edges = flip_edges
        self.edge_ratio_epsilon = 0.
        self.num_aug = 1


def read_seg(seg):
    seg_labels = np.loadtxt(open(seg, 'r'), dtype='float64')
    return seg_labels


def make_soft_eseg(obj_path, eseg_path, seseg_path, nclasses=4):
    if not os.path.isdir(seseg_path): os.makedirs(seseg_path)
    files = glob.glob(os.path.join(obj_path, '*.obj'))
    opt = BaseOptions()
    opt.initialize()
    opt.num_aug = 1

    for file in files:
        mesh = from_scratch(file, opt)
        gemm_edges = np.array(mesh.gemm_edges)
        obj_id = os.path.splitext(os.path.basename(file))[0]
        seg_file = os.path.join(eseg_path, obj_id + '.eseg')
        edge_seg = np.array(read_seg(seg_file).squeeze(), dtype='int32') - 1
        s_eseg = -1 * np.ones((mesh.edges_count, nclasses), dtype='float64')
        for ei in range(mesh.edges_count):
            prob = np.zeros(nclasses)
            seg_ids, counts = np.unique(edge_seg[gemm_edges[ei]], return_counts=True)
            prob[seg_ids] = counts / float(len(gemm_edges[ei]))
            s_eseg[ei, :] = prob
        s_eseg_file = os.path.join(seseg_path, obj_id + '.seseg')
        np.savetxt(s_eseg_file, s_eseg, fmt='%f')


def find_nearest_vector(array, value):
    idx = np.array([np.linalg.norm(x) for x in array-value]).argmin()
    return array[idx]


def get_obj_info_from_file(file):
    """
    解析obj文件，得到对应的vs和faces
    """
    vs, faces = [], []
    f = open(file)
    for line in f:
        line = line.strip()
        splitted_line = line.split()
        if not splitted_line:
            continue
        elif splitted_line[0] == 'v':
            vs.append([float(v) for v in splitted_line[1:4]])
        elif splitted_line[0] == 'f':
            face_vertex_ids = [int(c.split('/')[0]) for c in splitted_line[1:]]
            assert len(face_vertex_ids) == 3
            # 下标从0开始
            face_vertex_ids = [(ind - 1) if (ind >= 0) else (len(vs) + ind)
                               for ind in face_vertex_ids]
            faces.append(face_vertex_ids)
    f.close()
    vs = np.asarray(vs)
    faces = np.asarray(faces, dtype=int)
    assert np.logical_and(faces >= 0, faces < len(vs)).all()
    return vs, faces


if __name__ == "__main__":
    from vtkplotter import load, show

    # a = load('/home/heygears/work/Tooth/8AP1S/test_tooth_5000.obj').wireframe().c(('yellow'))
    # # b = load('/home/heygears/work/Tooth/8AP1S/test.vtk').pointSize(10).c(('green'))
    # # c = load('/home/heygears/work/Tooth/8AP1S/down.vtk').pointSize(8).c(('red'))
    #
    # d = load('/home/heygears/work/Tooth/8AP1S/file/mesh1.obj').c(('blue'))
    # e = load('/home/heygears/work/Tooth/8AP1S/file/mesh2.obj').c(('magenta'))
    #
    # show(a, d, e)

    a = load("/home/heygears/work/MeshCNN/datasets/tooth/stl/ADR2L_VS_SET_VSc3_Subsetup13_Maxillar.stl").wireframe().c(('pink'))
    b = load("/home/heygears/work/MeshCNN/datasets/tooth/obj/ADR2L_VS_SET_VSc3_Subsetup13_Maxillar.obj").wireframe().c(('red'))
    show(a, b)