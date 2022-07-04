from itertools import combinations_with_replacement

import torch
from torch.linalg import vector_norm


def dsh_loss(predictions, ground_truth_classes, margin, alpha) -> torch.tensor:

    # Preprocess the inputs for computing the loss (simulating the inputs for siamese network)
    comb = combinations_with_replacement(zip(predictions, ground_truth_classes), 2)
    comb = [(b1, b2, (y1 == y2).int()) for ((b1, y1), (b2, y2)) in comb]
    h1 = torch.stack([x[0] for x in comb])
    h2 = torch.stack([x[1] for x in comb])
    targets = torch.stack([x[2] for x in comb])

    # Deep Supervised Hashing (DSH) Loss
    # https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_Deep_Supervised_Hashing_CVPR_2016_paper.pdf

    # l2_dist == hamming_distance when h1 and h2 are perfect binary
    l2_dist = torch.square(vector_norm(h1 - h2, ord=2, dim=1))

    # Loss term for similar-pairs (i.e when target == 1)
    # It punishes similar images mapped to different binary codes
    l1 = 0.5 * targets * l2_dist

    # Loss term for dissimilar-pairs (i.e when target == 0)
    # It punishes dissimilar images mapped to close binary codes
    l2 = 0.5 * (1 - targets) * torch.max(margin - l2_dist, torch.zeros_like(l2_dist))

    # Regularization term
    l3 = alpha * (vector_norm(torch.abs(h1) - torch.ones_like(h1), ord=1, dim=1) +
                  vector_norm(torch.abs(h2) - torch.ones_like(h2), ord=1, dim=1))

    minibatch_loss = l1 + l2 + l3
    loss = torch.mean(minibatch_loss)
    return loss