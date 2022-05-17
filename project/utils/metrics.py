import torch


def compute_map_score(train_hash_codes, train_ground_truths, query_hash_codes, query_ground_truths):
    """
    credits: https://github.com/weixu000/DSH-pytorch/blob/906399c3b92cf8222bca838c2b2e0e784e0408fa/utils.py#L58
    :param train_hash_codes: An input tensor of dims (num_train, hash_length) with binary values
    :param train_ground_truths: An input tensor of dims (num_train,)
    :param query_hash_codes: An input tensor of dims (num_query, hash_length) with binary values
    :param query_ground_truths: An input tensor of dims (num_query,)
    :return: map_score
    """
    AP = []  # average precision
    num_samples = torch.arange(1, train_hash_codes.size(0) + 1)
    for i in range(query_hash_codes.size(0)):
        query_label, query_hash = query_ground_truths[i], query_hash_codes[i]
        hamming_dist_between_query_and_training_data = torch.sum((query_hash != train_hash_codes).long(), dim=1)
        ranking = hamming_dist_between_query_and_training_data.argsort()
        correct = (query_label == train_ground_truths[ranking]).float().to('cpu')
        P = torch.cumsum(correct, dim=0) / num_samples  # precision vector
        AP.append(torch.sum(P * correct) / torch.sum(correct))
    map_score = torch.mean(torch.Tensor(AP))
    return map_score
