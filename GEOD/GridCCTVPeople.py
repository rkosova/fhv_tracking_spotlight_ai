import torch

from torch.utils.data import Dataset
from torchvision.io import read_image

import os

from typing import Any

class GridCCTVPeople(Dataset):
    def __init__(
            self,
            annotations_dir,
            img_dir,
            transform=None,
            target_transform=None,
            grid_cells=16
    ) -> None:

        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        self.grid_cells = grid_cells

        self.labels = []
        self.paths = [os.path.splitext(i)[0] for i in os.listdir(annotations_dir)]
        self.paths.sort()

        for p in self.paths:
            _ = torch.zeros(grid_cells)
            
            with open(os.path.join(annotations_dir, p +'.txt'), 'r') as f:
                lines = f.readlines()
                for l in lines:
                    ohe = torch.zeros_like(_) 
                    ohe[int(l)] = 1
                    _ = torch.clamp((_ + ohe), max=1) # OHE tensors of objects in grid per image are added and clamped to 1
                                                      # As such, the final tensor represents a OHE of every object for each cell (not a count of objects per cell)
            self.labels.append(_)
    

    def __len__(self):
        return len(self.labels)


    def __getitem__(self, index) -> Any:
        image_path = os.path.join(self.img_dir, self.paths[index] + '.png')
        image = read_image(image_path).to(torch.float32)
        label = self.labels[index]

        return image, label # flattened label to one tensor of shape (sxs, ), each occurance encodes to 1 even if multiple objects in cell
