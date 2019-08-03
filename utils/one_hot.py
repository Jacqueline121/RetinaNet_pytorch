import torch


def one_hot_embedding(labels, class_num):
    '''
    Embedding labels to one-hot form
    :param labels: class labels, size: [N,]
    :param class_num: number of classes.
    :return: encoded label, size: [N, class_num]
    '''
    y = torch.eye(class_num)  # [D,D]
    return y[labels]            # [N,D]


if __name__ == '__main__':
    s = one_hot_embedding([9, 3], 20)
    print(s)