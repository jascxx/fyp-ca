import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint

'''
Modified code from https://github.com/locuslab/smoothing/blob/master/code/core.py
'''

class Smooth(object):
    """A smoothed classifier g (Modified for Anomaly Detection) """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, sigma: float):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch] -> Anomaly Score
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.sigma = sigma
        self.anomaly_scores = {}
        self.cache = {} # to cache (fpr, tpr, ar) given a threshold

    def predict(self, threshold: float, id: int, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).
        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.
        :param threshold: anomaly threshold
        :param id: id of tensor (used for caching)
        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(threshold, id, x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    
    def _sample_noise(self, threshold: float, id: int, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.
        :param threshold: anomaly threshold
        :param id: id of tensor (used for caching)
        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        scores = self._sample_anomaly_score(id, x, num, batch_size)
        normal_count = self._modified_binary_search(scores, threshold)
        return np.array([normal_count, len(scores) - normal_count])
        
    
    # arr is sorted. return number of elements such that arr[i] < val.
    def _modified_binary_search(self, arr, val):
        low, hi = 0, len(arr)
        if arr[hi - 1] < val:
            return len(arr)
        
        while (low < hi - 1):
            mid = (low + hi) // 2
            if arr[mid] >= val:
                hi = mid
            else:
                low = mid
        return low
        
    def _sample_anomaly_score(self, id: int, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's anomaly score under noisy corruptions of the input x.
        :param id: id of tensor (used for caching)
        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: a tensor containing the anomaly scores.
        """
        if id not in self.anomaly_scores:
            with torch.no_grad():
                predictions = []
                for _ in range(ceil(num / batch_size)):
                    this_batch_size = min(batch_size, num)
                    num -= this_batch_size
                    batch = x.repeat((this_batch_size, 1, 1, 1))
                    noise = torch.randn_like(batch, device=torch.device('cpu')) * self.sigma
                    pred = self.base_classifier.score(batch + noise) # pred is a 1d tensor of Anomaly Scores
                    predictions.append(pred)
                self.anomaly_scores[id], _ = torch.sort(torch.cat(predictions, 0))
                
        return self.anomaly_scores[id]


    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.
        This function uses the Clopper-Pearson method.
        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]

    '''
    Get the fpr, tpr, and abstain rate for a given threshhold.
    '''
    def get_fpr_tpr_ar(self, X, y_true, threshold):
        print('           threshold {:3f}'.format(threshold))
        if threshold not in self.cache:
            y_pred = []
            for i, x in enumerate(X):
                res = self.predict(threshold, i, x, 1000, 0.01, 256)
                y_pred.append(res)
            y_pred = np.array(y_pred)

            fp = np.sum((y_pred == 1) & (y_true == 0))
            tp = np.sum((y_pred == 1) & (y_true == 1))

            fn = np.sum((y_pred == 0) & (y_true == 1))
            tn = np.sum((y_pred == 0) & (y_true == 0))

            _fpr = fp / (fp + tn)
            _tpr = tp / (tp + fn)
            _ar = np.sum(y_pred == -1) / len(y_pred)
            
            self.cache[threshold] = (_fpr, _tpr, _ar)
        res = self.cache[threshold]
        print('fpr = {:3f}, tpr = {:3f}, ar = {:3f}'.format(res[0], res[1], res[2]))
        return res
    
    '''
    Function to get the roc curve. With randomized smoothing, Getting the roc curve is not trivial as 
    the smoothed classifiers are different for each threshold. This is because the smoothed classifier
    is defined based on label {0,1} and not the anomaly scores.
    '''
    def get_fprs_tprs_ars(self, X, y_true, low, high, max_diff=0.02):
        print('---------- threshold range {:3f}-{:3f} ----------'.format(low, high))
        low_fpr, low_tpr, low_ar = self.get_fpr_tpr_ar(X, y_true, low)
        high_fpr, high_tpr, high_ar = self.get_fpr_tpr_ar(X, y_true, high)

        if low_fpr - high_fpr < max_diff:
            return [high_fpr, low_fpr], [high_tpr, low_tpr], [high_ar, low_ar]

        mid = (low + high) / 2
        low_half_fprs, low_half_tprs, low_half_ars = self.get_fprs_tprs_ars(X, y_true, low, mid)
        high_half_fprs, high_half_tprs, high_half_ars = self.get_fprs_tprs_ars(X, y_true, mid, high)
        return high_half_fprs + low_half_fprs[1:], high_half_tprs + low_half_tprs[1:], high_half_ars + low_half_ars[1:]

