import torch


def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = torch.sqrt(linear_HSIC(X, X))
    var2 = torch.sqrt(linear_HSIC(Y, Y))
    return hsic / (var1 * var2)


def linear_HSIC(X, Y):
    L_X = X @ X.T
    L_Y = Y @ Y.T
    return torch.sum(centering(L_X) * centering(L_Y))


def centering(K):
    n = K.shape[0]
    unit = torch.ones([n, n]).to(K.device)
    I = torch.eye(n).to(K.device)
    H = I - unit / n

    return (H @ K) @ H  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
    # return np.dot(H, K)  # KH
