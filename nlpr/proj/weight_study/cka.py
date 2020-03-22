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


def center_gram(gram_matrix):
    """ Center a symmetric Gram matrix

    :param gram_matrix: torch.FloatTensor, an [N, N] Gram matrix
    :return: torch.FloatTensor, an [N, N] centered Gram matrix
    """
    means = gram_matrix.mean(dim=0)
    means -= means.mean() / 2
    return gram_matrix - means[:, None] - means[None, :]


def compute_linear_gram(x):
    """ Compute Gram matrix from activation matrix with a linear kernel

    :param x: torch.FloatTensor, the first [N, D] activation matrix
    :return: torch.FloatTensor, an [N, N] Gram matrix
    """
    return x @ x.t()


def compute_rbf_gram(x, threshold=1.0):
    """ Compute Gram matrix from activation matrix with RBF kernel

    :param x: torch.FloatTensor, the first [N, D] activation matrix
    :param threshold: Fraction of median Euclidean distance to use as RBF kernel bandwidth.
    :return: torch.FloatTensor, an [N, N] Gram matrix
    """
    dot_products = x @ x.t()
    sq_norms = torch.diag(dot_products)
    sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
    sq_median_distance = torch.median(sq_distances)
    return torch.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


def cka_from_gram(gram_x, gram_y):
    """ Compute CKA from Gram matrices

    :param gram_x: torch.FloatTensor, the first [N, N] Gram matrix
    :param gram_y: torch.FloatTensor, the second [N, N] Gram matrix
    :return: torch.FloatTensor, the CKA similarity
    """
    centered_gram_x = center_gram(gram_x)
    centered_gram_y = center_gram(gram_y)
    hsic = torch.sum(centered_gram_x * centered_gram_y)
    var1 = torch.sum(centered_gram_x * centered_gram_x)
    var2 = torch.sum(centered_gram_y * centered_gram_y)
    return hsic / torch.sqrt(var1 * var2)


def compute_cka(x, y, kernel="linear"):
    """ Compute CKA between activation matrices x and y

    :param x: torch.FloatTensor, the first [N, D] activation matrix
    :param y: torch.FloatTensor, the second [N, D] activation matrix
    :param kernel: "linear" or "rbf" for kernel
    :return: torch.FloatTensor, the CKA similarity
    """
    if kernel == "linear":
        gram_x, gram_y = compute_linear_gram(x), compute_linear_gram(y)
    elif kernel == "rbf":
        gram_x, gram_y = compute_rbf_gram(x), compute_rbf_gram(y)
    else:
        raise KeyError(kernel)
    return cka_from_gram(gram_x=gram_x, gram_y=gram_y)
