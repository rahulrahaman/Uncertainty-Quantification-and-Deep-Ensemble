import torch, torchvision
import torch.nn as nn
import os
import models
import utils.datautils as datutils
import argparse
import utils.trainutils as trainutil


def return_resnet34(nclass):
    resnet34 = torchvision.models.resnet34()
    resnet34.fc = nn.Linear(512, nclass).to(device)
    return resnet34


# SETUP GPU
if not os.path.exists('saved_models'):
    os.mkdir('saved_models')

torch.backends.cudnn.benchmark = True
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
download = True
model_dict = {'CIFAR10': models.FastResNet,
              'CIFAR100': models.FastResNet,
              'DIABETIC_RETINOPATHY': models.DRModel,
              'IMAGENETTE': return_resnet34(nclass=10),
              'IMAGEWOOF': return_resnet34(nclass=10),
              'MNIST': models.LeNet
              }

parser = argparse.ArgumentParser(description='Train multiple models sequentially')

parser.add_argument("--dataset", type=str, choices=['CIFAR10', 'CIFAR100', 'DIABETIC_RETINOPATHY', 'IMAGEWOOF',
                                                    'IMAGENETTE'], default='CIFAR10', help="Name of the dataset")
parser.add_argument("--datadir", type=str, help="Path to dataset")
parser.add_argument("--nmodel", type=int, help="How many models to train (Deep Ensemble)", default=1)
parser.add_argument("--mixup", type=float, help="Alpha for mixup, omit to train without mixup", default=None)
parser.add_argument("--ntrain", type=int, help="How many training example to include, -1 for full dataset", default=-1)
parser.add_argument("--nval", type=int, help="How many validation example to include", default=0)
parser.add_argument("--epoch", type=int, help="Number of epochs to train")
parser.add_argument("--max_lr", type=float, help="Maximum learning rate during LR scheduling (OneCycleLR)")
parser.add_argument("--bsize", type=int, help="Batch size")
parser.add_argument("--wd", type=float, help="Weight decay")

args = parser.parse_args()

# Dataset
print(f'Dataset chosen: {args.dataset}')
loaders, num_class = datutils.return_loaders(dataset=args.dataset, base=args.datadir, batch_size=args.bsize,
                                             start=args.ntrain, end=args.nval+args.ntrain)
# Architecture
model_def = model_dict[args.dataset]

for i in range(0, args.nmodel):
    model = model_def().to(device)
    optimizer, scheduler = trainutil.create_optim_schedule(model, loaders['train'], args.epoch, max_lr=args.max_lr,
                                                           weight_decay=args.wd, pct_start=0.25)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    last_accuracy = trainutil.perform_train(model=model, criterion=criterion, loaders=loaders, optimizer=optimizer,
                                            scheduler=scheduler, mixup=args.mixup, n_epoch=args.epoch, device=device)

    dataset = args.dataset.upper()
    savefile = dataset
    savefile += '_ntrain-' + str(len(loaders['train'].dataset))
    savefile += '_MixUpAlpha-' + str(args.mixup)
    savefile += '_id-' + str(i+1)
    checkpoint = {'model_state': model.state_dict(),
                  'optim_state': optimizer.state_dict(),
                  'acc': last_accuracy}
    torch.save(checkpoint, 'saved_models/' + savefile + '.model')

