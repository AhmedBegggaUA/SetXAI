import sys
sys.path.insert(0, '/Users/ahmedbegga/Desktop/TFG-Ahmed/SetXAI')
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *
from dspn import *
from utils import *
import numpy as np
import h5py
from torch.utils.data import DataLoader, Dataset, TensorDataset
from data.MnistSet import *
from time import sleep
from tqdm import tqdm
from data.pointcloud_utils import *
from data.Modelnet10toSet import *
import argparse
class Net(nn.Module):
    def __init__(self, set_encoder, set_decoder, input_encoder=None):
        """
        In the auto-encoder setting, don't pass an input_encoder because the target set and mask is
        assumed to be the input.
        In the general prediction setting, must pass all three.
        """
        super().__init__()
        self.set_encoder = set_encoder
        self.input_encoder = input_encoder
        self.set_decoder = set_decoder
        for m in self.modules():
            if (
                isinstance(m, nn.Linear)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Conv1d)
            ):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input, target_set, target_mask):
        if self.input_encoder is None:
            # auto-encoder, ignore input and use target set and mask as input instead
            latent_repr = self.set_encoder(target_set, target_mask)
            target_repr = latent_repr
        else:
            # set prediction, use proper input_encoder
            latent_repr = self.input_encoder(input)
            # note that target repr is only used for loss computation in training
            # during inference, knowledge about the target is not needed
            target_repr = self.set_encoder(target_set, target_mask)

        predicted_set = self.set_decoder(latent_repr)

        return predicted_set, (target_repr, latent_repr)
def main():
    #Procesamiento de los argumentos
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="FSEncoderDSPN", help="Encoder for set,[MaxEncoderDSPN,SumEncoderDSPN,MeanEncoderDSPN,FSEncoderDSPN]")
    parser.add_argument(
        "--batchsize", type=int, default=32, help="Batch size to train with"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-2, help="Outer learning rate of model"
    )
    parser.add_argument(
        "--dataset",
        default="mnist",
        choices=["mnist","modelnet10"],
        help="Use MNIST or ModelNet10 datasets",
    )
    parser.add_argument(
        "--loss",
        default="chamfer",
        choices=["chamfer","hungarian"],
        help="Use chamfer loss or hungarian loss",
    )
    parser.add_argument(
        "--epoch", type=int, default=100, help="Epochs for the model"
    )
    parser.add_argument(
        "--latent", type=int, default=None, help="Dimensionality of latent space"
    )
    parser.add_argument(
        "--dim", type=int, default=None, help="Dimensionality of hidden layers"
    )
    parser.add_argument(
        "--inner_lr", type=int, default=800, help="Inner learning rate"
    )
    parser.add_argument(
        "--iters", type=int, default=10, help="Inner iterations"
    )
    parser.add_argument("--store", action="store_true", help="")
    args = parser.parse_args()
    batch_size = args.batchsize
    if args.dataset == "mnist":
        train_loader = get_loader(
                    MNISTSet(train=True, full=True), batch_size=batch_size, num_workers=0)
        test_loader = get_loader(
                    MNISTSet(train=False, full=True), batch_size=batch_size, num_workers=0)
    elif args.dataset == "modelnet10":
        train_data = PointCloudData()
        test_data = PointCloudData(None, Train=False, folder='test', transform=None)
        train_loader = DataLoader(train_data, batch_size=32)
        test_loader = DataLoader(test_data, batch_size=32)
    if args.dataset == "mnist":
        set_channels = 2
        set_size = 342
        hidden_dim = 32
        latent_dim = 16
    else:
        set_channels = 3
        set_size = 2048
        hidden_dim = 128
        latent_dim = 64
    inner_lr = args.inner_lr
    iters = args.iters
    encoder = globals()[args.encoder]
    set_encoder = encoder(set_channels, latent_dim, hidden_dim)
    set_decoder = DSPN(set_encoder, set_channels, set_size, hidden_dim, iters, inner_lr)
    net = Net(
        input_encoder=None, set_encoder=set_encoder, set_decoder=set_decoder
    )
    n_epochs =args.epoch
    optimizer = torch.optim.Adam(
    [p for p in net.parameters() if p.requires_grad], lr=0.01)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Traning using ',device,' with ',args.encoder,'and dataset',args.dataset)
    net.train()
    print(net)
    losses = []
    accs = []
    for epoch in range(n_epochs):
        with tqdm(train_loader, unit="batch") as tepoch:
            for i, sample in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")
                if(device == 'gpu' and args.dataset == "mnist"):
                    label, set, target_mask = map(lambda x: x.cuda(), sample)
                elif args.dataset == "mnist":
                    label, set, target_mask = map(lambda x: x, sample)
                else:
                    label = sample['category']
                    set = sample['pointcloud']
                    target_mask = sample['mask']
                (progress, masks, evals, gradn), (y_enc, y_label) = net(label, set, target_mask)
                
                progress_only = progress
                set = torch.cat([set, target_mask.unsqueeze(dim=1)], dim=1)
                progress = [torch.cat([p, m.unsqueeze(dim=1)], dim=1)
                                for p, m in zip(progress, masks)]
                if args.loss == 'chamfer':
                    set_loss = chamfer_loss(torch.stack(progress), set.unsqueeze(0))
                else:
                    # dim 0 is over the inner iteration steps
                    # target set is broadcasted over dim 0
                    a = torch.stack(progress)
                    # target set is explicitly broadcasted over dim 0
                    b = set.repeat(a.size(0), 1, 1, 1)
                    # flatten inner iteration dim and batch dim
                    a = a.view(-1, a.size(2), a.size(3))
                    b = b.view(-1, b.size(2), b.size(3))
                    set_loss = hungarian_loss(
                    progress[-1], set
                    ).unsqueeze(0)
                loss = set_loss.mean()
                losses.append(loss.item())
                #accs.append(acc.item())
                progress = progress_only
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())
            print('loss: {}'.format(round(sum(losses)/len(losses),2)))
    if args.store:
            nombre =  args.encoder + "_model_" +args.dataset + ".pth"
            torch.save(net.state_dict(),nombre)
    print('TEST')
    net.eval()
    losses = []
    accs = []
    with tqdm(test_loader, unit="batch") as tepoch:
        for i, sample in enumerate(tepoch):
            tepoch.set_description(f"Test")
            if(device == 'gpu' and args.dataset == "mnist"):
                    label, set, target_mask = map(lambda x: x.cuda(), sample)
            elif args.dataset == "mnist":
                label, set, target_mask = map(lambda x: x, sample)
            else:
                label = sample['category']
                set = sample['pointcloud']
                target_mask = sample['mask']
            (progress, masks, evals, gradn), (y_enc, y_label) = net(label, set, target_mask)
                
            progress_only = progress
            set = torch.cat([set, target_mask.unsqueeze(dim=1)], dim=1)
            progress = [torch.cat([p, m.unsqueeze(dim=1)], dim=1)
                            for p, m in zip(progress, masks)]
            if args.loss == 'chamfer':
                    set_loss = chamfer_loss(torch.stack(progress), set.unsqueeze(0))
            else:
                # dim 0 is over the inner iteration steps
                # target set is broadcasted over dim 0
                a = torch.stack(progress)
                # target set is explicitly broadcasted over dim 0
                b = set.repeat(a.size(0), 1, 1, 1)
                # flatten inner iteration dim and batch dim
                a = a.view(-1, a.size(2), a.size(3))
                b = b.view(-1, b.size(2), b.size(3))
                set_loss = hungarian_loss(progress[-1], set).unsqueeze(0)
            loss = set_loss.mean()
            progress = progress_only
            tepoch.set_postfix(loss=loss.item())
            losses.append(loss.item())
    print('loss: {}'.format(round(sum(losses)/len(losses),2)))
if __name__ == '__main__':    
    main()