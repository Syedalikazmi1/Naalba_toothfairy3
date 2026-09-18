

import torch
import torch.nn as nn
import torch.nn.functional as F


from image_encoder import UNET
from point_embedder import get_embedder
from mlp import NeRF



class NeBLaModel(nn.Module):

    def __init__(self):

        super().__init__()


        # Image encoder
        self.image_encoder = UNET(
            in_channels=1,
            out_channels=128
        )


        # Official NeBLa point embedding
        self.embedder, self.input_ch = get_embedder(
            multires=7
        )


        # NeBLa NeRF MLP
        self.nerf = NeRF(
            D=8,
            W=128,
            input_ch=self.input_ch,
            output_ch=1
        )



    def sample_image_feature(
        self,
        feature_map,
        uv
    ):

        """
        feature_map:
            (B,128,H,W)

        uv:
            (N,2)
            normalized [-1,1]

        """

        B,C,H,W = feature_map.shape


        uv = uv.float()

        grid = uv.view(
            B,
            -1,
            1,
            2
        )


        feature_map = feature_map.float()

        sampled = F.grid_sample(
            feature_map,
            grid,
            align_corners=True
        )


        sampled = sampled.squeeze(-1)


        sampled = sampled.permute(
            0,
            2,
            1
        )


        return sampled



    def forward(
        self,
        image,
        points,
        uv
    ):

        """
        image:
            (1,1,H,W)

        points:
            (N,3)

        uv:
            (N,2)
            normalized coordinates

        """

        # encode image

        feature_map = self.image_encoder(
            image
        )



        # embed 3D points

        embedded_points = self.embedder(
            points
        )


        # sample image features

        image_features = self.sample_image_feature(
            feature_map,
            uv
        )


        image_features=image_features.reshape(
            -1,
            128
        )


        # NeRF density

        density = self.nerf(
            embedded_points,
            image_features
        )


        return density

