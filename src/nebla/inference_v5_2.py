
import torch
import numpy as np
import os
import sys

sys.path.append(
    "/content/drive/MyDrive/NeBLa_ToothFairy3_Project/src/nebla"
)

from ray_model import NeBLaRayModel
from renderer import volume_render


# =========================
# SETTINGS
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)


CHECKPOINT="/content/drive/MyDrive/NeBLa_ToothFairy3_Project/results/checkpoints/nebla_v5_2_multicase_best.pth"


DATA_ROOT="/content/drive/MyDrive/NeBLa_ToothFairy3_Project/data/training_dataset/final_nebla_dataset"


CASES=[
    "ToothFairy3F_001",
    "ToothFairy3F_002_0000",
    "ToothFairy3F_003_0000",
    "ToothFairy3F_004_0000"
]


OUTPUT="/content/drive/MyDrive/NeBLa_ToothFairy3_Project/results/inference_v5_2_multicase"

os.makedirs(
    OUTPUT,
    exist_ok=True
)


# =========================
# LOAD MODEL
# =========================

model=NeBLaRayModel().to(DEVICE)


model.load_state_dict(
    torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )
)


model.eval()


print("✅ Model loaded successfully")


results=[]


# =========================
# EVALUATION
# =========================

for case_name in CASES:

    print("\n======================")
    print("Evaluating:", case_name)
    print("======================")


    case_path=os.path.join(
        DATA_ROOT,
        case_name
    )


    simpx=np.load(
        case_path+"/simpx.npy"
    )


    rays=np.load(
        case_path+"/rays.npy"
    )


    print(
        "SIMPX:",
        simpx.shape,
        "RAYS:",
        rays.shape
    )


    image=torch.tensor(
        simpx,
        dtype=torch.float32,
        device=DEVICE
    )


    image=image.unsqueeze(0).unsqueeze(0)


    ray_points=torch.tensor(
        rays,
        dtype=torch.float32,
        device=DEVICE
    )


    predictions=[]


    with torch.no_grad():

        for i in range(ray_points.shape[0]):

            pts=ray_points[i]


            density=model(
                image,
                pts
            )


            rendered=volume_render(
                density,
                pts
            )


            predictions.append(
                rendered.cpu()
            )


            if i % 5000 == 0:
                print("Processed rays:", i)



    prediction=torch.stack(
        predictions
    ).numpy()



    target=simpx.reshape(-1)

    pred=prediction.reshape(-1)



    mse=np.mean(
        (pred-target)**2
    )


    psnr=10*np.log10(
        1/mse
    )


    print("----------------------")
    print("MSE:", mse)
    print("PSNR:", psnr)
    print("----------------------")


    np.save(
        OUTPUT+"/"+case_name+"_prediction.npy",
        prediction
    )


    results.append(
        [
            case_name,
            mse,
            psnr
        ]
    )


print("\n======================")
print("FINAL RESULTS")
print("======================")


for r in results:
    print(
        r[0],
        "MSE:",
        r[1],
        "PSNR:",
        r[2]
    )
