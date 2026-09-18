

import torch



def volume_render(
    density,
    ray_points
):

    """
    NeBLa volume rendering

    density:
        (N_samples,1)

    ray_points:
        (N_samples,3)

    returns:
        rendered pixel intensity
    """



    # distance between consecutive samples

    distances = torch.norm(
        ray_points[1:] - ray_points[:-1],
        dim=-1
    )


    # add last distance

    distances = torch.cat(
        [
            distances,
            distances[-1:]
        ]
    )


    density = density.squeeze(-1)



    alpha = 1.0 - torch.exp(
        -density * distances
    )


    # accumulated transmittance

    trans = torch.cumprod(
        torch.cat(
            [
                torch.ones(
                    1,
                    device=density.device
                ),

                1.0-alpha+1e-10
            ]
        ),
        dim=0
    )[:-1]



    weights = trans * alpha



    color = torch.sum(
        weights * density
    )



    return color.unsqueeze(0)

