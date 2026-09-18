
import sys
import os

NEBLA_SRC=os.path.dirname(os.path.abspath(__file__))

if NEBLA_SRC not in sys.path:
    sys.path.insert(0, NEBLA_SRC)



import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from ray_dataset import NeBLaRayDataset
from ray_model import NeBLaRayModel
from renderer import volume_render

from v5_paths import CASES, CHECKPOINT

import numpy as np



device="cuda"



# ============================
# Dataset
# ============================

dataset=NeBLaRayDataset(
    CASES
)



loader=DataLoader(
    dataset,
    batch_size=1,
    shuffle=True
)



# ============================
# Load all SIMPX images
# ============================

case_images=[]


for case in CASES:

    img=np.load(
        case["simpx"]
    )

    img=torch.tensor(
        img,
        dtype=torch.float32
    ).to(device)


    img=(
        img
        .unsqueeze(0)
        .unsqueeze(0)
    )


    case_images.append(img)



print(
    "Loaded SIMPX images:",
    len(case_images)
)



# ============================
# Model
# ============================


model=NeBLaRayModel().to(device)



optimizer=torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)



criterion=nn.MSELoss()



epochs=20


best_loss=float("inf")



print("==============================")
print("Starting NeBLa V5.1 Training")
print("Device:",device)
print("Cases:",len(CASES))
print("==============================")



for epoch in range(epochs):


    model.train()


    total_loss=0



    for batch in loader:


        ray_points=batch["ray_points"].to(device)

        target=batch["target"].float().to(device)


        case_id=batch["case_id"].item()



        image_tensor=case_images[case_id]



        density=model(

            image_tensor,

            ray_points.squeeze(0)

        )



        rendered=volume_render(

            density,

            ray_points.squeeze(0)

        )



        loss = criterion(
            rendered.reshape(-1,1),
            target
        )




        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



        total_loss += loss.item()



    epoch_loss=total_loss/len(loader)



    print(
        f"Epoch {epoch+1} Loss {epoch_loss}"
    )



    if epoch_loss < best_loss:


        best_loss=epoch_loss


        torch.save(

            model.state_dict(),

            CHECKPOINT

        )


        print(
            "Best checkpoint saved"
        )



print("==============================")
print("NeBLa V5.1 Training Complete")
print(CHECKPOINT)
print("==============================")

