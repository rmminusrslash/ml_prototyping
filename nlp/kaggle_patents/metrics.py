import numpy as np
from loguru import logger
from scipy import stats


class Pearsonr:
    """
    Computes pearsonr and can be directly passed to HuggingFace Trainer's `compute_metrics` argument
    to monitor evaluation metrics while training.
    """

    def __call__(self, eval_pred):
        print(eval_pred)
        logits, labels = eval_pred
        print(logits)
        predictions = logits.reshape(-1)
        score = stats.pearsonr(labels, predictions).statistic
        return {"pearsonr": score}
