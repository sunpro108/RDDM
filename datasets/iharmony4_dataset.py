import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
import torchvision.transforms.functional as tf


class IHarmony4Dataset(data.Dataset):
    """A template dataset class for you to implement custom datasets."""

    def __init__(self, dataset_dir, is_for_train, image_size=256):
        """Initialize this dataset class.
        Parameters:
        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        # super().__init__()
        self.dataset_dir = Path(dataset_dir)  
        self.image_paths, self.mask_paths, self.gt_paths = [], [], []
        self.is_train = is_for_train
        self.train_file = None
        self._load_images_paths()
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor() 
        ]) 

    def _load_images_paths(
        self,
    ):
        file_name = "IHD_train.txt" if self.is_train else "IHD_test.txt"
        self.trainfile = str(self.dataset_dir / file_name)
        with open(self.trainfile, "r") as f:
            for line in f.readlines():
                line = line.rstrip()
                # line = line.replace("jpg", "png")
                name_parts = line.split("_")
                mask_path = line.replace("composite_images", "masks")
                mask_path = mask_path.replace(("_" + name_parts[-1]), ".png")
                gt_path = line.replace("composite_images", "real_images")
                gt_path = gt_path.replace(
                    "_" + name_parts[-2] + "_" + name_parts[-1], ".jpg"
                )
                self.image_paths.append(str(self.dataset_dir / line))
                self.mask_paths.append(str(self.dataset_dir / mask_path))
                self.gt_paths.append(str(self.dataset_dir / gt_path))

    def __getitem__(self, index):
        comp = Image.open(self.image_paths[index]).convert("RGB")
        real = Image.open(self.gt_paths[index]).convert("RGB")
        mask = Image.open(self.mask_paths[index]).convert("RGB")

        comp = self.transform(comp)
        real = self.transform(real)
        mask = self.transform(mask)
        comp = self._compose(comp, mask, real)

        return [real, comp, mask] # gt, input, input_condition

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)

    def _compose(self, fore, mask, back):
        return fore * mask + back * (1 - mask)

    def _simplify_filepath(self, img_path):
        _names = img_path.split('/')
        dataset_name = _names[-3]
        file_name = _names[-1].split('.')[0]
        return f"{dataset_name}/{file_name}"

    def load_name(self, index, sub_dir=False):
        # condition
        name = self.image_paths[index]
        return os.path.basename(name)

    def get_pad_size(self, index, block_size=8):
        img = Image.open(self.image_paths[index])
        patch_size = self.image_size
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        bottom = 0
        right = 0
        if h < patch_size:
            bottom = patch_size-h
            h = patch_size
        if w < patch_size:
            right = patch_size-w
            w = patch_size
        bottom = bottom + (h // block_size) * block_size + \
            (block_size if h % block_size != 0 else 0) - h
        right = right + (w // block_size) * block_size + \
            (block_size if w % block_size != 0 else 0) - w
        return [bottom, right]