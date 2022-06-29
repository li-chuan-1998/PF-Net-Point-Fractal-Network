import torch.utils.data as data
import torch
import os
import open3d as o3d
import numpy as np

def read_pcd(filename):
    pcd = o3d.io.read_point_cloud(filename)
    return np.array(pcd.points)

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
    return pcd[idx[:n]]

class PartDataset(data.Dataset):
    def __init__(self, root, input_size=5000, output_size=16384, normalize=True):
        self.input_size = input_size
        self.output_size = output_size
        self.normalize = normalize
        self.cache = dict()
        self.complete_dir = os.path.join(root, "complete/")
        self.partial_dir = os.path.join(root, "partial/")

        for idx, partial_pcd in enumerate(os.listdir(self.partial_dir)):
            if idx == 1000:
                break
            partial_pcd_tensor = torch.from_numpy(resample_pcd(read_pcd(self.partial_dir+partial_pcd), self.input_size))

            name_com = partial_pcd.split("_")
            complete_pcd_name = "_".join(name_com[:3]) + "_complete.pcd"
            complete_pcd_tensor = torch.from_numpy(read_pcd(self.complete_dir+complete_pcd_name))

            # if self.normalize:
            #     point_set = self.pc_normalize(point_set)

            self.cache[idx] = (complete_pcd_tensor, partial_pcd_tensor)
            print(len(self.cache), end=" ")

    
    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        couple = self.cache[idx]
        return couple[0], couple[1]

    def pc_normalize(self, pc):
        """ pc: NxC, return NxC """
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
