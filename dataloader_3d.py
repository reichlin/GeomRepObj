import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class DatasetSim3D(Dataset):

    def __init__(self, path, pov=False):

        self.pov = pov

        np.random.seed(0)
        random.seed(0)

        n_dir = len(os.listdir(path+"z_0/"))
        for dir in range(n_dir):

            if dir == 0:
                self.img_0 = np.load(path + "img_0/0.npy")
                self.img_1 = np.load(path + "img_1/0.npy")
                self.img_2d = np.load(path + "2d_img_0/0.npy")
                self.pos_0_int = np.load(path + "z_0/0.npy")
                self.pos_0_ext = np.load(path + "h_0/0.npy")
                self.pos_1_int = np.load(path + "z_1/0.npy")
                self.pos_1_ext = np.load(path + "h_1/0.npy")
                if pov:
                    self.img_0_pov = np.load(path + "img_0_pov/0.npy")
                    self.img_1_pov = np.load(path + "img_1_pov/0.npy")
            else:
                self.img_0 = np.concatenate((self.img_0, np.load(path + "img_0/"+str(dir)+".npy")), 0)
                self.img_1 = np.concatenate((self.img_1, np.load(path + "img_1/"+str(dir)+".npy")), 0)
                self.img_2d = np.concatenate((self.img_2d, np.load(path + "2d_img_0/"+str(dir)+".npy")), 0)
                self.pos_0_int = np.concatenate((self.pos_0_int, np.load(path + "z_0/"+str(dir)+".npy")), 0)
                self.pos_0_ext = np.concatenate((self.pos_0_ext, np.load(path + "h_0/"+str(dir)+".npy")), 0)
                self.pos_1_int = np.concatenate((self.pos_1_int, np.load(path + "z_1/"+str(dir)+".npy")), 0)
                self.pos_1_ext = np.concatenate((self.pos_1_ext, np.load(path + "h_1/"+str(dir)+".npy")), 0)
                if pov:
                    self.img_0_pov = np.concatenate((self.img_0_pov, np.load(path + "img_0_pov/"+str(dir)+".npy")), 0)
                    self.img_1_pov = np.concatenate((self.img_1_pov, np.load(path + "img_1_pov/"+str(dir)+".npy")), 0)

        print("Dataset Loaded in RAM")

        self.T = self.img_0.shape[0]
        self.field_size = np.array([[8.85, 8.85]])
        self.field_offset = np.array([[4.4, 4.4]])

        self.rot_0 = np.repeat(np.expand_dims(self.pos_0_ext[:, 2], -1), 2, -1)
        self.pos_0_int = (self.pos_0_int + self.field_offset) / self.field_size * np.array([[1, 1]])
        self.pos_0_ext = (self.pos_0_ext[:, :2] + self.field_offset) / self.field_size * np.array([[1, 1]])
        self.pos_1_int = (self.pos_1_int + self.field_offset) / self.field_size * np.array([[1, 1]])
        self.pos_1_ext = (self.pos_1_ext[:, :2] + self.field_offset) / self.field_size * np.array([[1, 1]])

        self.pos_0_int = np.flip(np.array([[0, 1]]) + np.array([[1, -1]]) * self.pos_0_int, axis=-1)
        self.pos_0_ext = np.flip(np.array([[0, 1]]) + np.array([[1, -1]]) * self.pos_0_ext, axis=-1)
        self.pos_1_int = np.flip(np.array([[0, 1]]) + np.array([[1, -1]]) * self.pos_1_int, axis=-1)
        self.pos_1_ext = np.flip(np.array([[0, 1]]) + np.array([[1, -1]]) * self.pos_1_ext, axis=-1)

    def __len__(self):
        return self.T

    def __getitem__(self, idx):

        if self.pov:
            img = torch.from_numpy(self.img_0_pov[idx]).float()
            next_img = torch.from_numpy(self.img_1_pov[idx]).float()
        else:
            img = torch.from_numpy(self.img_0[idx]).float()
            next_img = torch.from_numpy(self.img_1[idx]).float()

        img_2d = torch.from_numpy(self.img_2d[idx]).float()
        a = torch.from_numpy(self.pos_1_int[idx]-self.pos_0_int[idx]).float()
        real_pos = torch.from_numpy(np.concatenate((self.pos_0_int[idx:idx+1], self.pos_0_ext[idx:idx+1], self.rot_0[idx:idx+1]), 0)).float()
        next_real_pos = torch.from_numpy(np.concatenate((self.pos_1_int[idx:idx+1], self.pos_1_ext[idx:idx+1], self.rot_0[idx:idx+1]), 0)).float()

        return img, img_2d, next_img, a, real_pos, next_real_pos

if __name__ == '__main__':

    path = "./3d_experiment/dataset/"
    dataloader = DatasetSim3D(path, pov=True)
    batch = dataloader.__getitem__(0)
