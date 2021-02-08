# Adaptive Label Smoothing for Classifier-based Mutual Information Neural Estimation


## Background

Estimating mutual information (MI) by neural networks has achieved significant practical success, especially in representation learning. Recent results further reduced the variance in the neural estimation by training a probabilistic classifier. However, the trained classifier tends to be overly confident about some of its predictions, which results in an overestimated MI that fails to capture the desired representation. 

To soften the classifier, we propose a novel scheme that smooths the label adaptively according to how extreme the probability estimates are. The resulting MI estimate is unbiased under only mild assumptions on the model.

## Experiments

The repository contains the implementation of the proposed adaptive label smoothing scheme. Experimental results on MNIST and CIFAR10 datasets confirmed that method yields better representation and achieves higher classification test accuracy among existing approaches in self-supervised representation learning. The results are reproducible using the jupyter notebooks in this repository [https://github.com/wangxuuu/AdapLS_MINE](https://github.com/wangxuuu/AdapLS_MINE):

- AdapLS.ipynb: The proposed adaptive label smoothing classifier-based model.
- PCM.ipynb: The probabilisitic classifier method.
- InfoNCE.ipynb: The noise contrastive method.

