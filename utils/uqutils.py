import numpy as np
from torch.nn import CrossEntropyLoss as CE
import torch


def prob_power_t(prob, t):
    """
    Raise probability vector to the power of t
    :param prob: (numpy.array) Probability vector of any dimension, last dimension needs to be classes
    :param t: (float) power to raise to
    :return: (numpy.array) prob like array
    """
    raised_prob = prob**t
    if type(raised_prob) == torch.Tensor:
        raised_prob = raised_prob / torch.sum(prob, dim=-1, keepdim=True)
    else:
        raised_prob = raised_prob / np.sum(prob, axis=-1, keepdim=True)
    return raised_prob


def calculate_ece(probs, correct, nbin=30, fn=abs):
    """
    Calculate ECE
    :param probs: (numpy.array) probability predictions (max probability)
    :param correct: (numpy.array) indicator whether prediction was true
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


def get_all_scores(probs, targets, nbin=30, fn=abs):
    """
    Calculate accuracy, ECE, negative log-likelihood, Brier score
    :param probs: (numpy.array) predictions of dimension N x C where N is number of example, C is classes
    :param targets: (numpy.array) targets of dimension N
    :param nbin: (int) number of bins for calculating ECE
    :param fn: (function) function to transform conf - acc to fn(conf - acc) for ECE, sECE
    :return: tuple containing Accuracy, ECE, NLL, Brier
    """
    preds = np.argmax(probs, axis=1)
    correct = (preds == targets)
    acc = np.mean(correct)
    class_probs = np.take_along_axis(probs, targets.astype(np.uint8)[:, None], axis=1)
    nll = np.mean(-np.log(class_probs))
    maxprobs = np.max(probs, axis=-1)
    ece = calculate_ece(maxprobs, correct, nbin=nbin, fn=fn)[-1]
    one_hot = np.eye(probs.shape[1])[targets.astype(np.int32)]
    brier_score = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    return acc, ece, nll, brier_score


def perform_tempscale(probs, targets):
    """
    Run a linear search for temperature and find the optimal temperature
    :param probs: (numpy.array / torch.tensor) probability of dimension N x C. for ensemble, send pooled probability
    :param targets: (numpy.array / torch.tensor) targets of dimension N
    :return: (float) best temperature
    """
    if type(probs) != torch.Tensor:
        probs = torch.tensor(probs)
    if type(targets) != torch.Tensor:
        targets = torch.tensor(targets)
    temperatures = np.exp(np.linspace(start=-3, stop=3, num=61, endpoint=True))
    losses = np.array([CE(t*torch.log(probs), targets).item() for t in temperatures])
    temperatures = temperatures[np.isnan(losses)]
    losses = losses[np.isnan(losses)]
    best_temp = temperatures[np.argmin(losses)]
    return best_temp

