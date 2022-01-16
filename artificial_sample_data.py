import torch

# T = 2 periods, M = 3 individual stocks, K = 8 fundamental characteristics
# T x M x K
X = torch.tensor(
    [
        # t=0
        [
                # stock 1
                [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
                # stock 2
                [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
                # stock 3
                [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
        ],
        # t=1
        [
                # stock 1
                [3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8],
                # stock 2
                [4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8],
                # stock 3
                [5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8]
        ]
    ],
    dtype=torch.float, requires_grad=False)

X_r = torch.tensor(
    [
        # t=0
        [
            # stock 1, stock 2, stock 3
            [1], [2], [3]
        ],
        # t=1
        [
            # stock 1, stock 2, stock 3
            [-1], [-2], [-3]
        ]
    ],
    dtype=torch.float, requires_grad=False
)

X_rt = torch.transpose(X_r, 0, 1)

# T = 2 periods (batch size); N = 2 test portfolios
# T x N
R = torch.tensor(
    [
        # t=0
        [
            # test pf 1, test pf 2
            1, 2
        ],
        # t=1
        [
            # test pf 1, test pf 2
            3, 4
        ]
    ],
    dtype=torch.float, requires_grad=False)



softmax_test = torch.tensor(
        [[[ 0.1483, -0.3250],
         [ -2.0, -0.3250],
         [ 0.1280, -0.3203]],

        [[ 0.1204, -0.3140],
         [ 0.1192, -0.3078],
         [ 0.1200, 0.5]]]
)

# a = torch.max(softmax_test, torch.tensor([0.]))
# b = torch.min(softmax_test, torch.tensor([0.]))
# a = torch.where(softmax_test > 0, softmax_test, torch.tensor(float('nan')))
# b = torch.where(softmax_test < 0, softmax_test, torch.tensor(float('nan')))
# print(a)
# print(b)
softmax = torch.nn.Softmax(dim=1)
# a = softmax(softmax_test)
# print(a)
# a = a - (1 / softmax_test.shape[1]) # not sure how to implemnt the softmax separation between the long and the short portfolio
# print(a)

# y_pos = softmax(a)
# print(y_pos)
# y_neg = softmax(b)
# print(y_neg)
# print(y_pos - y_neg)