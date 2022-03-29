from ast import arg
from json import encoder
import sys
from xmlrpc.client import Boolean
sys.path.insert(0, "/Users/ahmedbegga/Desktop/TFG-Ahmed/SetXAI")
from operator import ne
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tensorboardX import SummaryWriter
from utils import chamfer_loss
from fspool import FSPool
from model import *
from MnistSet import MNISTSet
from MnistSet import get_loader
from time import sleep
from tqdm import tqdm
from data.pointcloud_utils import *
from data.Modelnet10toSet import *
def main():
    #Procesando los parametros pasados por terminal
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="FSEncoder", help="Encoder for set,[MaxEncoder,SumEncoder,MeanEncoder,FSEncoder]")
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
        "--epoch", type=int, default=50, help="Epochs for the model"
    )
    
    parser.add_argument(
        "--latent", type=int, default=16, help="Dimensionality of latent space"
    )
    parser.add_argument(
        "--dim", type=int, default=32, help="Dimensionality of hidden layers"
    )
    parser.add_argument(
        "--train-only", action="store_true", help="Only run training, no evaluation"
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="Only run evaluation, no training"
    )
    parser.add_argument("--store", action="store_true", help="")
    args = parser.parse_args()
    writer = SummaryWriter(f"runs/{args.encoder}", purge_step=0)
    if args.dataset == "mnist":
        set_channels = 2
        set_size = 342
    else:
        set_channels = 3
        set_size = 2048
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
    lr = args.lr
    n_epochs = args.epoch
    hidden_dim = args.dim
    latent_dim = args.latent
    set_encoder = globals()[args.encoder]
    net = set_encoder(set_channels,latent_dim,hidden_dim)
    ##CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Traning using ',device,' with ',args.encoder,'and dataset',args.dataset)
    net = net.to(device)
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=0.001)
    #MNIST
    if not args.eval_only :
        net.train()
        net = net.double()
        for epoch in range(n_epochs):
            with tqdm(train_loader, unit="batch") as tepoch:
                for i, sample in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}")
                    if(device == 'gpu'):
                        label, target_set, target_mask = map(lambda x: x.cuda(), sample)
                    else:
                        label, target_set, target_mask = map(lambda x: x, sample)
                    optimizer.zero_grad()
                    output = net(target_set,target_mask)
                    loss = F.cross_entropy(output, label)
                    acc = (output.max(dim=1)[1] == label).float().mean()
                    writer.add_scalar("metric/loss", loss, global_step=i)
                    writer.add_scalar("metric/acc", acc.item(), global_step=i)
                    loss.backward()
                    optimizer.step()
                    tepoch.set_postfix(loss=loss.item(), acc=100. * acc.item())
        writer.close()
        if args.store:
            nombre =  args.encoder + "_model_" +args.dataset + ".pth"
            torch.save(net.state_dict(),nombre)
    if not args.train_only:
        net.eval()
        with tqdm(test_loader, unit="batch") as tepoch:
            for i, sample in enumerate(tepoch):
                tepoch.set_description(f"Test")
                if(device == 'gpu'):
                        input, target_set, target_mask = map(lambda x: x.cuda(), sample)
                else:
                    input, target_set, target_mask = map(lambda x: x, sample)
                output = net(target_set,target_mask)
                loss = F.cross_entropy(output, input)
                acc = (output.max(dim=1)[1] == input).float().mean()
                tepoch.set_postfix(loss=loss.item(), acc=100. * acc.item())



if __name__ == '__main__':    
    main()