import torch


def old_linear_CKA(X, Y):
    hsic = old_linear_HSIC(X, Y)
    var1 = torch.sqrt(old_linear_HSIC(X, X))
    var2 = torch.sqrt(old_linear_HSIC(Y, Y))
    return hsic / (var1 * var2)


def old_linear_HSIC(X, Y):
    L_X = X @ X.T
    L_Y = Y @ Y.T
    return torch.sum(old_centering(L_X) * old_centering(L_Y))


def old_centering(K):
    n = K.shape[0]
    unit = torch.ones([n, n]).to(K.device)
    I = torch.eye(n).to(K.device)
    H = I - unit / n

    # HKH are the same with KH, KH is the first centering, H(KH) do the second time,
    # results are the sme with one time centering
    return (H @ K) @ H
    # return np.dot(H, K)  # KH


def center_gram(k):
    means = k.mean(dim=0)
    means -= means.mean() / 2
    return k - means[:, None] - means[None, :]


def compute_linear_gram(x):
    return x @ x.t


def compute_rbf_gram(x, threshold=1.0):
    dot_products = x @ x.T
    sq_norms = torch.diag(dot_products)
    sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
    sq_median_distance = torch.median(sq_distances)
    return torch.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


def cka_from_gram(gram_x, gram_y):
    centered_gram_x = center_gram(gram_x)
    centered_gram_y = center_gram(gram_y)
    hsic = torch.sum(centered_gram_x * centered_gram_y)
    var1 = torch.sum(centered_gram_x * centered_gram_x)
    var2 = torch.sum(centered_gram_y * centered_gram_y)
    return hsic / torch.sqrt(var1 * var2)


def compute_cka(x, y, kernel="linear"):
    if kernel == "linear":
        gram_x, gram_y = compute_linear_gram(x), compute_linear_gram(y)
    elif kernel == "rbf":
        gram_x, gram_y = compute_rbf_gram(x), compute_rbf_gram(y)
    else:
        raise KeyError(kernel)
    return cka_from_gram(gram_x=gram_x, gram_y=gram_y)
