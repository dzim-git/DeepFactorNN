"""
Author: David Zimmermann
E-Mail: david.zimmermann@greenmail.ch
Date created: 10.01.2022
Version: 0.1
Python version: 3.9

This module is a potential partial sample implementation of the neural network used in the paper "Deep Learning in
Characteristics-Sorted Factor Models" by Guanhao Feng, Nicholas G. Polson, Jianeng Xu.
URL: https://arxiv.org/abs/1805.01104

It takes as input the time series of K fundamental stock characteristics of M individual stocks named X, as well as
the stock's returns named X_r to predict the returns of N test portfolios called R.
The network builds P "deep-factor" portfolios from the individual stocks and estimates the test portfolios' sensitivity
to these factor portfolio returns (betas) to maximize the model's explanatory power for the test portfolio returns.

In the original paper the authors combined the deep factors with the Fama French factor portfolios, which has been
omitted because of its relative triviality. A more interesting step would be the addition of a GAN to build the
test portfolios from the input stocks, as used in "Deep Learning in Asset Pricing" (Luyang Chen, Markus Pelgery and
Jason Zhuz): https://economics.yale.edu/sites/default/files/deep_learning_in_asset_pricing.pdf

TODO: 1. connect with real data
      2. create data loader to load batches and separate into training, validation and test sets
      3. expand with GAN
"""

import torch
import tqdm
from matplotlib import pyplot as plt


class DeepFactorNN(torch.nn.Module):

    def __init__(self, n_stock_characteristics: int, n_test_pf: int, n_deep_factors: int = 2):
        super(DeepFactorNN, self).__init__()
        self.lin1 = torch.nn.Linear(n_stock_characteristics, 4)  # K x 4 ; K original fundamental stock characteristics, applied repetitively over M individual stocks and batches T
        self.lin2 = torch.nn.Linear(4, 4)                        # 4 x 4 ; hidden layer
        self.lin3 = torch.nn.Linear(4, n_deep_factors)           # 4 x P ; P = 2 deep factors
        self.softmax = torch.nn.Softmax(dim=1)
        self.betas = torch.nn.Parameter(torch.rand(n_test_pf, n_deep_factors), requires_grad=True)   # N x P ; N = 2 test pf, P = 2 deep factors (betas)

    def forward(self, x: torch.Tensor, X_r: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.lin1(x))
        x = torch.tanh(self.lin2(x))
        x = torch.tanh(self.lin3(x))

        # calculate portfolio weights of deep factor portfolios
        # TODO: not sure how to implement the softmax separation between the long and the short portfolio
        x = self.softmax(x) - (1 / x.shape[1])      # T x M x P

        # calculate deep factor portfolio returns
        x = torch.transpose(x, 1, 2)                # T x M x P   -->   T x P x M
        x = torch.matmul(x, X_r)                    # T x P x M   mmult  T x M x 1    -->  T x P x 1
        x = x.squeeze()                             # reduce dimensionality T x P x 1  -->  T x P

        # calculate expected test portfolio return
        betas_t = torch.transpose(self.betas, 0, 1)     # N x P   -->   P x N
        R_hats = torch.matmul(x, betas_t)           # T x P   mmult   P x N   -->   T x N
        return R_hats


def target_function(r_hat: torch.Tensor, R: torch.Tensor, lamda: float):
    assert r_hat.shape == R.shape, "Ground truth and estimates do not have the same dimensions"
    diffs = R - r_hat

    # calculate pricing errors
    t_avg_squared = torch.square(torch.mean(diffs, dim=0))
    pricing_error = torch.mean(t_avg_squared)

    # calculate time series variation
    squared_diff = torch.square(diffs)
    ts_var = torch.mean(squared_diff)

    # lamda weighted loss
    loss = ts_var + lamda * pricing_error   # tensor of size 1 (float)
    return loss


def train_nn(lamda: float, n_deep_factors: int, n_epochs: int) -> torch.nn.Module:
    # TODO: add loader for real data instead of
    # load data
    from artificial_sample_data import X, X_r, R
    n_test_pf: int = R.shape[1]
    n_stock_characteristics: int = X.shape[2]

    # create NN
    nn = DeepFactorNN(n_stock_characteristics=n_stock_characteristics, n_test_pf=n_test_pf,
                      n_deep_factors=n_deep_factors)
    optimizer = torch.optim.Adam(nn.parameters())

    # training
    hist_loss = []
    print("start training for {} epochs".format(n_epochs))
    for _ in tqdm.tqdm(range(n_epochs)):
        # TODO: add loop for batches with real data
        r_hat = nn.forward(X, X_r)
        loss = target_function(r_hat, R, lamda)
        hist_loss.append(float(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("done training. Final training loss: {}".format(float(loss)))

    plot_training_loss(hist_loss)
    return nn


def plot_training_loss(hist_loss):
    plt.plot(hist_loss)
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.title('Loss per epoch')
    plt.show()


if __name__ == '__main__':
    LAMDA = 0.5
    N_DEEP_FACTORS = 2
    N_EPOCHS = 1000
    nn = train_nn(LAMDA, N_DEEP_FACTORS, N_EPOCHS)