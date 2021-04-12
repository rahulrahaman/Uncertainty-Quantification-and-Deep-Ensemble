import numpy as np
import torch
import time as time


def mixup_data(x, y, alpha=1.0, device=None):
    """
    Returns augmented image lam*x1 + (1-lam)*x2 by mixing x1 and x2
    with coefficient lam simulated from beta(alpha, alpha)
    :param x: (torch.tensor) images
    :param y: (torch.tensor) labels
    :param alpha: (float) parameter of the beta distribution
    :param device: (torch.device) device to put the augmented image
    :return: (augmented image, label1:label of x1, label2:label of x2, lambda:simulated lambda)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if device is not None:
        index = torch.randperm(batch_size).to(device)
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    loss function for augmented image x = lam*x_a + (1-lam)*x_b
    :param criterion: (torch.nn.criterion) this simplistic formula works for CrossEntropy-like losses
    :param pred: (torch.tensor) prediction for augmented image x
    :param y_a: (torch.tensor) label of x_a
    :param y_b: (torch.tensor) label of x_b
    :param lam: (float) mixing coefficient of x_a and x_b
    :return: loss:torch.tensor
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(model, dataloader, device, optimizer, scheduler, criterion, epoch, mixup=None):
    """
    train one epoch
    :param model: (torch.nn.Module) model to be trained
    :param dataloader: (torch.utils.data.DataLoader) dataloader
    :param device: (torch.device) device to put tensors into
    :param optimizer: (torch.nn.optimizer) optimizer for model
    :param scheduler: (torch.nn.scheduler) scheduler to control lr
    :param criterion: (torch.nn.criterion) loss criterion
    :param epoch: (int) which epoch is running
    :param mixup: (float or None) the alpha parameter for beta distribution to generate mixup
    :return: dict with average loss value and compute time
    """
    model.train()
    train_loss_list = []
    start = time.time()
    for batch in dataloader:
        inputs, targets = batch[0].to(device), batch[1].to(device)

        if mixup is not None:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, mixup, device)

        outputs = model(inputs)
        if mixup is not None:
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            loss = criterion(outputs, targets)

        train_loss_list.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update learning rate
        if scheduler is not None:
            scheduler.step()

    compute_time = time.time() - start
    avg_loss = np.mean(np.array(train_loss_list))
    return {"avg_loss": avg_loss,
            "compute_time": compute_time}


def evaluate(model, dataloader, criterion, device, evalmode=True):
    """
    run inference after an epoch finishes
    :param model: (torch.nn.Module) model to be trained
    :param dataloader: (torch.utils.data.DataLoader) dataloader
    :param criterion: (torch.nn.criterion) loss criterion
    :param device: (torch.device) device to put tensors into
    :param evalmode: (bool) whether to use model.eval while running inference
    :return: dict with values as average loss and accuracy in the test set
    """
    correct = 0
    total = 0
    total_loss = 0.
    if evalmode:
        model.eval()
    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch[0].to(device), batch[1].to(device)
            pred = model(images)
            # LOSS
            loss = criterion(pred, labels)
            total_loss += loss.item() * labels.size(0)
            # ACCURACY
            predicted = torch.argmax(pred.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / float(total)
    avg_loss = total_loss / float(total)
    return {"accuracy": accuracy,
            "avg_loss": avg_loss}


def get_lr(optimizer):
    """
    Get current learning rate and momentum from optimizer
    :param optimizer: (torch.nn optimizer)
    :return: dict with values lr and momentum
    """
    for param_group in optimizer.param_groups:
        return {"lr": param_group['lr'],
                "momentum": param_group.get("momentum", 0)}


def create_optim_schedule(model, trainloader, n_epoch, max_lr, pct_start=.25, weight_decay=5e-4):
    """
    Return optimizer and lr scheduler
    :param model: (torch.nn.Module) model to be trained
    :param trainloader: (torch.utils.data.DataLoader) dataloader for training dataset
    :param n_epoch: (int) number of epochs to train
    :param max_lr: (float) maximum learning rate during scheduling
    :param pct_start: (0.0 < float < 1.0) pct_start for OneCycleLRScheduler, when to stop increasing lr
    :param weight_decay: (float) weight decay for optimizer
    :return: optimizer, scheduler
    """
    params = []
    if isinstance(model, list):
        for mod in model:
            params += list(mod.parameters())
    else:
        params = model.parameters()
        
    optimizer = torch.optim.SGD(params,
                                lr=max_lr,
                                momentum=0.9,
                                weight_decay=weight_decay,
                                nesterov=True)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=max_lr,
                                                    epochs=n_epoch,
                                                    steps_per_epoch=len(trainloader),
                                                    pct_start=pct_start,
                                                    anneal_strategy='linear',
                                                    cycle_momentum=False,
                                                    div_factor=100.0,  # 25
                                                    final_div_factor=10000.0,
                                                    last_epoch=-1)
    return optimizer, scheduler


def perform_train(model, criterion, loaders, optimizer, scheduler, mixup, n_epoch, device, trainfn=train,
                  trainon='train'):
    """

    :param model: (torch.nn.Module) model to be trained
    :param loaders: list(torch.utils.data.DataLoader) list of train dataloader, test and validation
    :param device: (torch.device) device to put tensors into
    :param optimizer: (torch.nn.optimizer) optimizer for model
    :param scheduler: (torch.nn.scheduler) scheduler to control lr
    :param criterion: (torch.nn.criterion) loss criterion
    :param n_epoch: (int) number of training epochs
    :param mixup: (float or None) the alpha parameter for beta distribution to generate mixup
    :param trainfn: (function) function to run one training epoch
    :param trainon: (str) the key for loaders dictionary, on which training is going to be performed
    :return: (dict) performance of model on test dataset after last epoch
    """

    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []

    total_train_time = 0
    criterion.to(device)

    dataloader_test = loaders['test']
    dataloader_trainnoaugment = loaders['no-augment_train']

    for epoch in range(n_epoch):
        train_results = trainfn(model, loaders[trainon], device, optimizer, scheduler, criterion, epoch, mixup)
        total_train_time += train_results["compute_time"]
        evaluate_test = evaluate(model, dataloader_test, criterion, device)
        evaluate_train = evaluate(model, dataloader_trainnoaugment, criterion, device)

        optim_param = get_lr(optimizer)

        print("Epoch: [{epoch}/{n_epoch}]  "
              "Loss: {train_loss:.2f}({test_loss:.2f})  "
              "Acc: {train_acc:.1f}%({test_acc:.1f}%)  "
              "Training: {total_train_time:.1f}sec ({compute:.1f}sec)  "
              "lr/momentum: {lr:.4f}/{momentum:.2f}".format(
                                                            epoch=epoch + 1, n_epoch=n_epoch,
                                                            train_loss=evaluate_train["avg_loss"],
                                                            test_loss=evaluate_test["avg_loss"],
                                                            train_acc=100 * evaluate_train["accuracy"],
                                                            test_acc=100 * evaluate_test["accuracy"],
                                                            total_train_time=total_train_time,
                                                            compute=train_results["compute_time"],
                                                            lr=optim_param["lr"],
                                                            momentum=optim_param["momentum"],))

        # save statistics for later plotting
        train_loss_list.append(evaluate_train["avg_loss"])
        test_loss_list.append(evaluate_test["avg_loss"])
        train_acc_list.append(evaluate_train["accuracy"])
        test_acc_list.append(evaluate_test["accuracy"])

    return test_acc_list[-1]


def calculate_ece(probs, correct, nbin=30, fn=abs):
    """
    Calculate ECE
    :param probs: (np.array) probability predictions (max probability)
    :param correct: (np.array) indicator whether prediction was true
    :param nbin: (int) number of bins for calculating ECE
    :param fn: (function) function to transform conf - acc to fn(conf - acc) for ECE, sECE
    :return: (float) ece
    """
    bins = (probs*nbin).astype(np.int)
    ece_total = np.array([np.sum(bins == i) for i in range(nbin+1)])
    ece_correct = np.array([np.sum((bins == i)*correct) for i in range(nbin+1)])
    acc = np.array([ece_correct/ece_total if ece_total > 0 else -1])
    conf = np.array([np.mean(probs[bins == i]) for i in range(nbin+1)])
    deviation = np.array([fn(acc[i] - conf[i]) if acc[i] >= 0 else 0 for i in range(nbin+1)])
    return ece_total, ece_correct, deviation

