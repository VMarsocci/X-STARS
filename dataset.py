import json
import numpy as np
import os
import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder


class CustomImageFolder(ImageFolder):
    def __init__(
        self,
        root,
        transform=None,
        msad_transform=None,
        target_transform=None,
        sensors=None,
    ):
        super(CustomImageFolder, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        self.root = root
        self.msad_transform = msad_transform
        self.sel_sensors = sensors
        self.av_sensors = ["SPOT", "Sentinel", "Landsat"]
        self.ex_sensors = list(set(self.av_sensors) - set(sensors))
        print("AVAILABLE SENSORS: ", self.av_sensors)
        print("SELECTED SENSORS: ", self.sel_sensors)
        self.normalizations = {
            "Landsat": transforms.Normalize(
                [0.27059479, 0.27839213, 0.18060363], [0.1741122, 0.14797395, 0.125955]
            ),
            "Sentinel": transforms.Normalize(
                [0.14316194, 0.14518686, 0.09228685],
                [0.13242126, 0.10325284, 0.08168219],
            ),
            "SPOT": transforms.Normalize(
                [0.31007201, 0.34869021, 0.23991865],
                [0.16708196, 0.14294321, 0.16296565],
            ),
        }

    def __getitem__(self, index):
        batch = {}
        path, target = self.imgs[index]

        sensor = path.split("/")[-3]
        img_name = os.path.basename(path)
        img = Image.open(path).convert("RGB")

        # Load image from other directory with the same name
        if self.msad_transform:
            oth_av_sensors = list(set(self.sel_sensors) - set(sensor))
            batch[f"{sensor}_noaugm"] = self.normalizations[sensor](
                self.msad_transform(img)
            )
            for oth_sensor in oth_av_sensors:
                oth_im_path = path.replace(sensor, oth_sensor)
                oth_im = Image.open(oth_im_path).convert("RGB")
                batch[f"{oth_sensor}_noaugm"] = self.normalizations[oth_sensor](
                    self.msad_transform(oth_im)
                )

        if self.transform is not None:
            img = self.transform(img)

        batch["samples"] = img
        batch["enc_sensor"] = torch.tensor(target)
        batch["sensor"] = sensor

        return batch
