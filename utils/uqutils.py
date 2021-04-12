import numpy as np


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
    :param nbin: (int) number of biins for calculating ECE
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


