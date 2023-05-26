import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize
import numpy as np
import math
import cv2
import configargparse


class ImageDataset(Dataset):
    def __init__(self,
                 image_path: str,
                 sidelength: list = None) -> None:
        super().__init__()
        self.image = cv2.imread(image_path)
        if sidelength != None:
            self.image = cv2.resize(self.image, tuple(sidelength))
        self.sidelength = self.image.shape[1::-1]

        self.image = self.normalize(self.image)
        self.coords, self.rgb = self.split(self.image)

    def normalize(self, image) -> torch.Tensor:
        transform = Compose([ToTensor(),
                             Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))])

        image = transform(image)
        image = image.permute(1, 2, 0)

        return image

    def split(self, image: torch.Tensor):
        H, W, C = image.shape
        [h, w] = torch.meshgrid(torch.linspace(-1, 1, H),
                                torch.linspace(-1, 1, W))
        y = h.contiguous().view(H, W, 1)
        x = w.contiguous().view(H, W, 1)
        rgb = image.view(H, W, C)
        coords = torch.cat([x, y], -1)
        return coords, rgb


def calc_psnr(img_path1: str, img_path2: str) -> float:
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    mse = np.mean((img1/255 - img2/255) ** 2)

    return -10 * math.log10(mse)


def config() -> configargparse.argparse.Namespace:
    args = configargparse.ArgumentParser()
    args.add("--config", required=False, is_config_file=True,
             help="Path to config file.")

    args.add_argument("--sidelength", nargs=2, type=int)
    args.add_argument("--seed", type=int, default=4233919)

    args.add_argument("--model_type", required=True, type=str,
                      help="model type")
    args.add_argument("--hash_table_length", nargs='+', type=int)
    args.add_argument("--in_features", type=int)
    args.add_argument("--out_features", type=int)
    args.add_argument("--hidden_layers", type=int)
    args.add_argument("--hidden_features", type=int)
    args.add_argument("--outermost_linear", type=bool, default=True)
    args.add_argument("--wf", type=int)
    args.add_argument("--wh", type=int)
    args.add_argument("--hash_mod", type=bool, default=True)

    args.add_argument("--lr", type=float, default=1e-4,
                      help="learning rate. default=1e-4")
    args.add_argument("--epochs", type=int, default=10000,
                      help="Number of iterations to train for. default=1e4")
    args.add_argument("--save_mod_path", type=str, required=True)
    args.add_argument("--img_path", type=str, required=True)

    args = args.parse_args()

    torch.random.manual_seed(args.seed)

    return args
