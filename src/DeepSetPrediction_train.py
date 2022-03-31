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
from data.Modelnet10toSet import *


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
    train_data = PointCloudData()
    test_data = PointCloudData(None, Train=False, folder='test', transform=None)
    train_loader = DataLoader(train_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)
    def build_net():
        set_channels = 3
        set_size = 2048
        hidden_dim = 256
        inner_lr = 800
        iters = 10
        latent_dim = 64
        input_encoder = None
        set_encoder = MaxEncoderDSPN(set_channels, latent_dim, hidden_dim)
        set_decoder = DSPN(set_encoder, set_channels, set_size, hidden_dim, iters, inner_lr)
        net = Net(
            input_encoder=input_encoder, set_encoder=set_encoder, set_decoder=set_decoder
        )
        return net
    net = build_net()
    n_epochs = 100
    optimizer = torch.optim.Adam(
    [p for p in net.parameters() if p.requires_grad], lr=0.01)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Traning using ',device,' with ','fspool','and dataset','pointcloud')
    net.train()
    print(net)
    for epoch in range(1):
        with tqdm(train_loader, unit="batch") as tepoch:
            for i, sample in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")
                if(device == 'gpu'):
                    label, set, target_mask = map(lambda x: x.cuda(), sample)
                else:
                    label, set, target_mask = map(lambda x: x, sample)
                (progress, masks, evals, gradn), (y_enc, y_label) = net(label, set, target_mask)
                
                progress_only = progress
                set = torch.cat([set, target_mask.unsqueeze(dim=1)], dim=1)
                progress = [torch.cat([p, m.unsqueeze(dim=1)], dim=1)
                                for p, m in zip(progress, masks)]
                set_loss = chamfer_loss(torch.stack(progress), set.unsqueeze(0))
                loss = set_loss.mean()
                progress = progress_only
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())
    print('TEST')
    net.eval()
    losses = []
    accs = []
    with tqdm(test_loader, unit="batch") as tepoch:
        for i, sample in enumerate(tepoch):
            tepoch.set_description(f"Test")
            if(device == 'gpu'):
                    label, set, target_mask = map(lambda x: x.cuda(), sample)
            else:
                label, set, target_mask = map(lambda x: x, sample)
            (progress, masks, evals, gradn), (y_enc, y_label) = net(label, set, target_mask)
            progress_only = progress
            set = torch.cat([set, target_mask.unsqueeze(dim=1)], dim=1)
            progress = [torch.cat([p, m.unsqueeze(dim=1)], dim=1)
                            for p, m in zip(progress, masks)]
            set_loss = chamfer_loss(torch.stack(progress), set.unsqueeze(0))
            loss = set_loss.mean()
            progress = progress_only
            tepoch.set_postfix(loss=loss.item())
            losses.append(loss.item())
    print('loss: {}, acc: {}'.format(round(sum(losses)/len(losses),2),round(sum(accs)/len(accs),2)))
if __name__ == '__main__':    
    main()