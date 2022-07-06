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
    def __init__(self, root, input_size=4096, output_size=16384, normalize=True):
        self.output_size = output_size
        self.cache = dict()
        self.complete_dir = os.path.join(root, "complete/")
        self.partial_dir = os.path.join(root, "partial/")

        total_size = len(os.listdir(self.complete_dir))
        for idx, complete_pcd in enumerate(os.listdir(self.partial_dir)):
            # partial_pcd_np = resample_pcd(read_pcd(self.partial_dir+partial_pcd), input_size)
            # complete_pcd_name = "_".join(partial_pcd.split("_")[:3]) + "_complete.pcd"
            complete_pcd_np = read_pcd(self.complete_dir+complete_pcd)

            if normalize:
                # partial_pcd_np = self.pc_normalize(partial_pcd_np)
                complete_pcd_np = self.pc_normalize(complete_pcd_np)


            # partial_pcd_tensor = torch.FloatTensor(partial_pcd_np)
            complete_pcd_tensor = torch.FloatTensor(complete_pcd_np)

            self.cache[idx] = complete_pcd_tensor
            # self.cache[idx] = (complete_pcd_tensor, partial_pcd_tensor)
            if len(self.cache) % 3000 == 0:
                print(len(self.cache), "pcds loaded | ", f"{total_size-idx} pcds left...")

    
    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        # couple = self.cache[idx]
        return self.cache[idx], None

    def pc_normalize(self, pc):
        """ pc: NxC, return NxC """
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
