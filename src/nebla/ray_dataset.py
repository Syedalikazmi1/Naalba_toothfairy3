

import torch
from torch.utils.data import Dataset
import numpy as np



class NeBLaRayDataset(Dataset):


    def __init__(self,cases):


        self.rays=[]
        self.targets=[]
        self.pixels=[]
        self.case_ids=[]


        for idx,case in enumerate(cases):


            print("Loading case:",idx)


            simpx=np.load(
                case["simpx"]
            )


            rays=np.load(
                case["rays"]
            )


            H,W=simpx.shape


            for i in range(
                rays.shape[0]
            ):


                y=i//W
                x=i%W


                self.rays.append(
                    rays[i]
                )


                self.targets.append(
                    simpx[y,x]
                )


                self.pixels.append(
                    [y,x]
                )


                self.case_ids.append(
                    idx
                )



        self.rays=torch.tensor(
            np.array(self.rays),
            dtype=torch.float32
        )


        self.targets=torch.tensor(
            np.array(self.targets),
            dtype=torch.float32
        ).unsqueeze(1)



        self.pixels=torch.tensor(
            np.array(self.pixels),
            dtype=torch.long
        )


        self.case_ids=torch.tensor(
            np.array(self.case_ids),
            dtype=torch.long
        )


        print("======================")
        print("NeBLa V5.1 Dataset")
        print("Total rays:",len(self.rays))
        print("Rays:",self.rays.shape)
        print("Targets:",self.targets.shape)
        print("======================")



    def __len__(self):

        return len(self.rays)



    def __getitem__(self,index):


        return {

            "ray_points":
            self.rays[index],


            "target":
            self.targets[index],


            "pixel":
            self.pixels[index],


            "case_id":
            self.case_ids[index]

        }

