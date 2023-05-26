import torch
import torch.nn as nn
import cv2
import numpy as np
from model import MLP, Siren, HashINR
from utils import config, calc_psnr

@torch.no_grad()
def render_img(model: nn.Module, save_path: str, sidelength: list):
    W, H = sidelength
    [h, w] = torch.meshgrid(torch.linspace(-1, 1, H),
                            torch.linspace(-1, 1, W))
    y = h.contiguous().view(H, W, 1)
    x = w.contiguous().view(H, W, 1)
    coords = torch.cat([x, y], -1)
    rgb = (model(coords) + 1) / 2
    # rgb = model.hash_table
    # ze = torch.zeros(sidelength).unsqueeze(dim=-1)
    # rgb = torch.cat([rgb,ze],dim=2)

    image = np.round(rgb.detach().numpy()*255).astype(np.uint8)

    cv2.imwrite(save_path, image)



def render(args):
    sidelength = args.sidelength

    model_type = args.model_type
    hash_table_length = args.hash_table_length
    in_features = args.in_features
    out_features = args.out_features
    hidden_layers = args.hidden_layers
    hidden_features = args.hidden_features
    outermost_linear = args.outermost_linear
    first_omega_0 = args.wf
    hidden_omega_0 = args.wh

    save_mod_path = args.save_mod_path
    img_path = args.img_path
    output_image = "output/recon.png"

    if model_type == 'Siren':
        model = Siren(
            in_features=in_features,
            out_features=out_features,
            hidden_layers=hidden_layers,
            hidden_features=hidden_features,
            outermost_linear=outermost_linear,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0
        )
    elif model_type == 'MLP':
        model = MLP(
            in_features=in_features,
            out_features=out_features,
            hidden_layers=hidden_layers,
            hidden_features=hidden_features
        )
    elif model_type == 'HashINR':
        model = HashINR(
            hash_table_length=hash_table_length,
            in_features=in_features,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=out_features,
            outermost_linear=True,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0
        )
    else:
        raise NotImplementedError("Model type not supported!")

    dic:dict = torch.load(save_mod_path)
    model.load_state_dict(dic)

    render_img(model, output_image, sidelength)
    psnr = calc_psnr(img_path, output_image)
    print("recon PSNR:{:.2f}".format(psnr))


if __name__ == "__main__":
    args = config()
    render(args)
