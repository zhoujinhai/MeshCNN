import os
import torch
import sys
sys.path.append("/home/heygears/work/MeshCNN/")
from data.base_dataset import BaseDataset
from util.util import is_mesh_file, pad
import numpy as np
from models.layers.mesh import Mesh


class SegmentationData(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.paths = self.make_dataset(self.dir)
        if opt.phase == "test":
            self.seg_paths = None
            self.sseg_paths = None
        else:
            self.seg_paths = self.get_seg_files(self.paths, os.path.join(self.root, 'seg'), seg_ext='.eseg')
            self.sseg_paths = self.get_seg_files(self.paths, os.path.join(self.root, 'sseg'), seg_ext='.seseg')
        self.classes, self.offset = self.get_n_segs(os.path.join(self.root, 'classes.txt'), self.seg_paths)
        self.nclasses = len(self.classes)
        self.size = len(self.paths)
        self.get_mean_std()
        # # modify for network later.
        opt.nclasses = self.nclasses
        opt.input_nc = self.ninput_channels

    def __getitem__(self, index):
        path = self.paths[index]
        # print("opt: {}, export_folder: {}, phase: {}".format(self.opt, self.opt.export_folder,  self.opt.phase))
        mesh = Mesh(file=path, opt=self.opt, hold_history=True, export_folder=self.opt.export_folder, phase=self.opt.phase)
        meta = {}
        meta["filename"] = mesh.filename
        meta['mesh'] = mesh
        if self.opt.phase != "test":
            label = read_seg(self.seg_paths[index]) - self.offset
            label = pad(label, self.opt.ninput_edges, val=-1, dim=0)
            meta['label'] = label
            soft_label = read_sseg(self.sseg_paths[index])
            meta['soft_label'] = pad(soft_label, self.opt.ninput_edges, val=-1, dim=0)

        # get edge features
        edge_features = mesh.extract_features()
        if edge_features.shape[1] <= self.opt.ninput_edges:
            edge_features = pad(edge_features, self.opt.ninput_edges)
            meta['edge_features'] = (edge_features - self.mean) / self.std
        else:
            meta["edge_features"] = np.array([])

        # print("edge_features: ", type(meta["edge_features"]), meta["edge_features"].shape)

        return meta

    def __len__(self):
        return self.size

    @staticmethod
    def get_seg_files(paths, seg_dir, seg_ext='.seg'):
        segs = []
        for path in paths:
            segfile = os.path.join(seg_dir, os.path.splitext(os.path.basename(path))[0] + seg_ext)
            assert(os.path.isfile(segfile))
            segs.append(segfile)
        return segs

    @staticmethod
    def get_n_segs(classes_file, seg_files):
        if not os.path.isfile(classes_file):
            all_segs = np.array([], dtype='float64')
            for seg in seg_files:
                all_segs = np.concatenate((all_segs, read_seg(seg)))
            segnames = np.unique(all_segs)
            np.savetxt(classes_file, segnames, fmt='%d')
        classes = np.loadtxt(classes_file)
        offset = classes[0]
        classes = classes - offset
        return classes, offset

    @staticmethod
    def make_dataset(path):
        meshes = []
        assert os.path.isdir(path), '%s is not a valid directory' % path

        for root, _, fnames in sorted(os.walk(path)):
            for fname in fnames:
                if is_mesh_file(fname):
                    path = os.path.join(root, fname)
                    meshes.append(path)

        return meshes


def read_seg(seg):
    seg_labels = np.loadtxt(seg, dtype='float64')
    return seg_labels


def read_sseg(sseg_file):
    sseg_labels = read_seg(sseg_file)
    sseg_labels = np.array(sseg_labels > 0, dtype=np.int32)
    return sseg_labels


if __name__ == "__main__":
    from options.train_options import TrainOptions
    print("1")
    from data import DataLoader
    opt = TrainOptions().parse()
    opt.dataroot = "/home/heygears/work/MeshCNN/" + opt.dataroot

    dataset = DataLoader(opt)
    dataset_size = len(dataset)
    print('#training meshes = %d' % dataset_size)