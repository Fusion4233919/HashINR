import torch
from torch import optim, nn
from model import MLP, Siren, HashINR
from utils import ImageDataset, config, calc_psnr
from render import render_img
from tqdm.autonotebook import tqdm


def train(args):

    if args.sidelength:
        sidelength = args.sidelength
    else:
        sidelength = None

    model_type = args.model_type
    hash_table_length = args.hash_table_length
    in_features = args.in_features
    out_features = args.out_features
    hidden_layers = args.hidden_layers
    hidden_features = args.hidden_features
    outermost_linear = args.outermost_linear
    first_omega_0 = args.wf
    hidden_omega_0 = args.wh

    lr = args.lr
    epochs = args.epochs
    save_mod_path = args.save_mod_path
    img_path = args.img_path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    model = model.to(device)
    optimizer = optim.Adam(lr=lr, params=model.parameters())
    loss = nn.MSELoss()

    Dataset = ImageDataset(img_path, sidelength)

    sidelength = Dataset.sidelength
    x, y = Dataset.coords.to(device), Dataset.rgb.to(device)

    with tqdm(range(epochs)) as bar:
        model.train()
        for epoch in bar:
            optimizer.zero_grad()
            logits = model(x)
            loss_mse = loss(logits, y)
            loss_mse.backward()
            optimizer.step()

            bar.desc = "PSNR:{:.2f} loss:{:.8f}".format(
                -10*torch.log10(loss_mse), loss_mse)

    torch.save(model.state_dict(), save_mod_path)

    if torch.cuda.is_available():
        model = model.to("cpu")
    render_img(model, "output/recon.png", sidelength)

    # psnr = calc_psnr(img_path, "output/recon.png")
    # print("recon PSNR:{:.2f}".format(psnr))


if __name__ == "__main__":
    args = config()
    train(args)
