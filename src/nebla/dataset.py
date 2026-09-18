

import torch
from torch.utils.data import Dataset
import numpy as np


class NeBLaToothFairyDataset(Dataset):

    def __init__(
        self,
        simpx,
        volume,
        samples_per_item=512
    ):

        self.simpx = torch.tensor(
            simpx,
            dtype=torch.float32
        )

        self.volume = volume.astype(
            np.float32
        )

        self.samples_per_item = samples_per_item

        self.shape = volume.shape


    def __len__(self):

        return 100


    def __getitem__(self, idx):

        Z,Y,X = self.shape


        # random voxel sampling

        x = np.random.randint(
            0,
            X,
            self.samples_per_item
        )

        y = np.random.randint(
            0,
            Y,
            self.samples_per_item
        )

        z = np.random.randint(
            0,
            Z,
            self.samples_per_item
        )


        points = np.stack(
            [
                x,
                y,
                z
            ],
            axis=1
        ).astype(
            np.float32
        )


        target = self.volume[
            z,
            y,
            x
        ]


        target = target[:,None]



        # projection coordinates

        u = x/(X-1)

        v = y/(Y-1)


        uv = np.stack(
            [
                u,
                v
            ],
            axis=1
        )


        uv = uv*2-1



        return {

            "image":
            self.simpx[None,:,:],


            "points":
            torch.tensor(points),


            "uv":
            torch.tensor(uv),


            "target":
            torch.tensor(target)

        }

