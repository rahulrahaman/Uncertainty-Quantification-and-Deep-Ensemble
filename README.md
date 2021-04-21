# Uncertainty Quantification and Deep Ensemble
##### Experiments from our work Uncertainty Quantification and Deep Ensemble https://arxiv.org/abs/2007.08792


In order to train a Deep Ensemble, use the python file `train_ensemble.py`. To get details of the command line arguments use the command `python train_ensemble.py --h`.
This will give the detailed usage of the arguments
```
Train multiple models sequentially

optional arguments:
  -h, --help            show this help message and exit
  --dataset {CIFAR10,CIFAR100,DIABETIC_RETINOPATHY,IMAGEWOOF,IMAGENETTE}
                        Name of the dataset
  --datadir DATADIR     Path to dataset
  --nmodel NMODEL       How many models to train (Deep Ensemble)
  --mixup MIXUP         Alpha for mixup, omit to train without mixup
  --ntrain NTRAIN       How many training example to include, -1 for full
                        dataset
  --nval NVAL           How many validation example to include
  --epoch EPOCH         Number of epochs to train
  --max_lr MAX_LR       Maximum learning rate during LR scheduling
                        (OneCycleLR)
  --bsize BSIZE         Batch size
  --wd WD               Weight decay

```
For example, if you need to train a Deep Ensemble on CIFAR10 dataset with 1000 training samples and mixup with alpha 0.5, use the command
```
python train_ensemble.py --dataset CIFAR10 --nmodel 5 --mixup 0.5 --ntrain 1000 --epoch 300 --max_lr 0.05 --bsize 500 --wd 7e-4 --datadir /home/data/
```
