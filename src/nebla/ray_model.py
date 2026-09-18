

import torch
import torch.nn as nn

from image_encoder import UNET
from mlp import NeRF
from point_embedder import get_embedder

from renderer import volume_render



class NeBLaRayModel(nn.Module):

    def __init__(self):

        super().__init__()


        # image encoder

        self.image_encoder = UNET(
            in_channels=1,
            out_channels=128,
            features=[32,64,128,256]
        )


        # positional embedding

        self.embedder, self.input_ch = get_embedder(
            multires=7
        )


        # NeBLa MLP

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


        B,C,H,W = feature_map.shape


        grid = uv.view(
            B,
            -1,
            1,
            2
        )


        sampled = torch.nn.functional.grid_sample(
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
        ray_points
    ):


        device = image.device


        # image features

        features = self.image_encoder(
            image
        )



        N = ray_points.shape[0]


        # normalize xyz to uv

        uv = ray_points[:,:2]


        uv = uv.clone()


        uv[:,0] = uv[:,0]/256
        uv[:,1] = uv[:,1]/256


        uv = uv*2-1



        # get feature for every point

        uv = uv.unsqueeze(0)


        image_feature = self.sample_image_feature(
            features,
            uv
        )


        image_feature=image_feature.squeeze(0)



        # embed points

        points_embedded = self.embedder(
            ray_points
        )


        density = self.nerf(
            points_embedded,
            image_feature
        )


        return density

