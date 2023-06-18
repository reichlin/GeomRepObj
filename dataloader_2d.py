import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import rotate


class DatasetSim2D(Dataset):

    def __init__(self, frq=0.5, N=100, move_bg=0, segment=0, batch_size=32, rot=True):

        self.frq = frq
        self.N = N
        self.batch_size = batch_size
        self.segment = segment
        self.rot = rot

        self.move_bg = move_bg
        if self.move_bg == 1:
            r_bg = 20
            self.bg = np.zeros((3 * N, 3 * N))
            for i in range(3 * N):
                for j in range(3 * N):
                    if int(i / r_bg) % 2 == 0 and int(j / r_bg) % 2 == 0:
                        self.bg[i, j] = 1
        else:
            self.bg = np.zeros((3 * N, 3 * N))

        self.r_g = 4
        self.r1 = 16 #8  # 2

        self.img_grip = np.zeros((9, 9))
        self.img_grip[0, 3:6] = 1
        self.img_grip[1, 3:6] = 1
        self.img_grip[2:4, :] = 1
        self.img_grip[4, 1:8] = 1
        self.img_grip[5, 2:7] = 1
        self.img_grip[6, 1:8] = 1
        self.img_grip[7, :] = 1
        self.img_grip[7, 4] = 0
        self.img_grip[8, :3] = 0
        self.img_grip[8, 6:] = 0

        self.img_obj1 = np.zeros((5, self.r1 * 2 + 1, self.r1 * 2 + 1))
        # diamond object
        for i in range(-int(self.r1/2), int(self.r1/2) + 1):
            for j in range(-int(self.r1/2), int(self.r1/2) + 1):
                if np.abs(i) + np.abs(j) <= int(self.r1/2):
                    self.img_obj1[0, i + int(self.r1/2), j + int(self.r1/2)] = 1

        # rectangular object
        for i in range(-int(self.r1 / 8) + 1, int(self.r1 / 8)):
            for j in range(-self.r1, self.r1 + 1):
                self.img_obj1[1, i + self.r1, j + self.r1] = 1

        for i in range(-self.r1, self.r1 + 1):
            for j in range(-int(self.r1 / 8) + 1, int(self.r1 / 8)):
                self.img_obj1[2, i + self.r1, j + self.r1] = 1

        self.img_obj1[3] = (rotate(self.img_obj1[1], angle=45, reshape=False) > 0.25) * 1.
        self.img_obj1[4] = (rotate(self.img_obj1[1], angle=-45, reshape=False) > 0.25) * 1.

    def __len__(self):
        return self.batch_size * 10

    def __getitem__(self, idx):

        state_t, state_t1, a = self.gen_interaction()

        img = torch.from_numpy(self.get_img(state_t["p"], state_t["p1"], state_t["r"])).float()
        next_img = torch.from_numpy(self.get_img(state_t1["p"], state_t1["p1"], state_t1["r"])).float()
        a = torch.from_numpy(a).float() / self.N
        rot_vals = np.array([[state_t["r"], state_t1["r"]]])
        real_pos = torch.from_numpy(np.concatenate((np.expand_dims(state_t["p"], 0), np.expand_dims(state_t["p1"], 0), rot_vals), 0)).float() / self.N
        next_real_pos = torch.from_numpy(np.concatenate((np.expand_dims(state_t1["p"], 0), np.expand_dims(state_t1["p1"], 0)), 0)).float() / self.N

        return img, img, next_img, a, real_pos, next_real_pos

    def gen_interaction(self):

        pos_obj1 = np.random.randint(self.r1 + self.r_g + 1, high=self.N - self.r1 - 1, size=2)
        rot_obj1 = np.random.randint(1, high=5) if self.rot else 0

        if np.random.random_sample() < self.frq:
            if self.segment:
                pos_grip = np.random.randint(self.r_g, high=self.N - self.r_g - 1, size=2)
                placed = False
                img = np.zeros((1, self.N, self.N))
                img = self.plot_element(img, pos_obj1, self.r1, self.img_obj1[rot_obj1], 0)
                while not placed:
                    pos_grip_1 = np.random.randint(self.r_g, high=self.N - self.r_g - 1, size=2)
                    for t in range(100):
                        cen = pos_grip + (t / 100.) * (pos_grip_1 - pos_grip)
                        for i in range(-self.r_g, self.r_g + 1):
                            for j in range(-self.r_g, self.r_g + 1):
                                if img[0, int(cen[0]) + i, int(cen[1]) + j] == 1:  # touch
                                    placed = True
                pos_obj1_1 = np.random.randint(self.r1 + self.r_g + 1, high=self.N - self.r1 - 1, size=2)
                rot_obj1_1 = np.random.randint(1, high=5) if self.rot else 0
            else:
                placed = False
                img = np.zeros((1, self.N, self.N))
                img = self.plot_element(img, pos_obj1, self.r1, self.img_obj1[rot_obj1], 0)
                while not placed:
                    pos_grip = np.random.randint(self.r_g, high=self.N - self.r_g - 1, size=2)
                    for i in range(-self.r_g, self.r_g + 1):
                        for j in range(-self.r_g, self.r_g + 1):
                            if img[0, int(pos_grip[0]) + i, int(pos_grip[1]) + j] == 1:
                                placed = True
                pos_grip_1 = np.random.randint(self.r_g, high=self.N - self.r_g - 1, size=2)
                pos_obj1_1 = np.random.randint(self.r1 + self.r_g + 1, high=self.N - self.r1 - 1, size=2)
                rot_obj1_1 = np.random.randint(1, high=5) if self.rot else 0
        else:
            pos_grip = np.random.randint(self.r_g, high=self.N - self.r_g - 1, size=2)
            pos_grip_1 = np.random.randint(self.r_g, high=self.N - self.r_g - 1, size=2)
            pos_obj1_1 = pos_obj1.copy()
            rot_obj1_1 = rot_obj1

        a = pos_grip_1 - pos_grip

        state_t = {"p": pos_grip, "p1": pos_obj1, "r": rot_obj1}
        state_t1 = {"p": pos_grip_1, "p1": pos_obj1_1, "r": rot_obj1_1}

        return state_t, state_t1, a

    def get_img(self, grip_pos, obj1_pos, rot):

        img = np.zeros((3, self.N, self.N))

        img[2] += self.bg[grip_pos[0] * 2:grip_pos[0] * 2 + self.N, grip_pos[1] * 2:grip_pos[1] * 2 + self.N]

        img = self.plot_element(img, grip_pos, self.r_g, self.img_grip, 0)
        img = self.plot_element(img, obj1_pos, self.r1, self.img_obj1[rot], 1)

        return img.copy()

    def plot_element(self, img, pos, r, template, channel):
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                img[:, int(pos[0]) + i, int(pos[1]) + j] *= (1 - template[i + r, j + r])
                img[channel, int(pos[0]) + i, int(pos[1]) + j] += template[i + r, j + r]
        return img


if __name__ == '__main__':
    dataloader = DatasetSim2D(frq=0.5, N=100, move_bg=1, segment=0, batch_size=32, rot=True)

    for i in range(100):
        batch = dataloader.__getitem__(i)
        print()


























