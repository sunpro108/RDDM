# evaluate PSNR of two group images in two directories.
import os
import sys
from pathlib import Path 

from tqdm import tqdm
from PIL import Image
import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision import transforms as T
from torchmetrics.image import PeakSignalNoiseRatio as PSNR

class Dataset(Dataset):
    def __init__(
        self,
        gen_folder,
        gt_folder,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
    ):
        super().__init__()
        self.gen_folder = gen_folder
        self.gt_folder = gt_folder
        self.gen_paths = [p for ext in exts for p in Path(f'{gen_folder}').glob(f'**/*.{ext}')]
        self.gt_paths = [p for ext in exts for p in Path(f'{gt_folder}').glob(f'**/*.{ext}')]

        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.gen_paths)

    def __getitem__(self, index):
        gen_path = self.gen_paths[index]
        gen_img = Image.open(gen_path).convert("RGB")
        gt_path = self.gt_paths[index]
        gt_img = Image.open(gt_path).convert("RGB")
        return self.transform(gen_img), self.transform(gt_img)

def f2i(image):
    # float -> uint8
    image = 255 * torch.clip(image, 0, 1.0)
    image = image.to(torch.uint8)
    return image


if __name__ == "__main__":

    if len(sys.argv)>1:
        gen_img_dir = sys.argv[1]
    else:
        gen_img_dir = "results_train_on_istd_1/test_timestep_10_10_pt"

    psnr = PSNR(data_range=(0.0, 255.0), reduction = 'elementwise_mean', dim = (1,2,3)).cuda()

    # predict images
    img_gen = Dataset(gen_folder=gen_img_dir,
                    gt_folder='home_datasets/ISTD_Dataset/test/test_C')

    for i, (gen_img, gt_img) in enumerate(tqdm(DataLoader(img_gen, batch_size=16))):
        gen_img, gt_img = gen_img.cuda(), gt_img.cuda()
        # print(gen_img.shape, gt_img.shape)
        gen_img = f2i(gen_img)
        gt_img = f2i(gt_img)
        psnr.update(gen_img, gt_img)

    print(psnr.compute().item())
