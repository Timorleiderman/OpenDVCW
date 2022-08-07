import random
import numpy as np


def self_information(p):
    return -np.log2(p)


def entropy(p):
    entropy = - p * np.log2(p)
    # Operator `nansum` will sum up the non-nan number
    out = np.nansum(entropy)
    return out


def cross_entropy(y_hat, y):
    ce = -np.log(y_hat[range(len(y_hat)), y])
    return np.mean(ce)


def joint_entropy(p_xy):
    joint_ent = -p_xy * np.log2(p_xy)
    # Operator `nansum` will sum up the non-nan number
    out = np.nansum(joint_ent)
    return out


def conditional_entropy(p_xy, p_x):
    p_y_given_x = p_xy/p_x
    cond_ent = -p_xy * np.log2(p_y_given_x)
    # Operator `nansum` will sum up the non-nan number
    out = np.nansum(cond_ent)
    return out


def mutual_information(p_xy, p_x, p_y):
    p = p_xy / (p_x * p_y)
    mutual = p_xy * np.log2(p)
    # Operator `nansum` will sum up the non-nan number
    out = np.nansum(mutual)
    return out


def kl_divergence(p, q):
    kl = p * np.log2(p / q)
    out = np.nansum(kl)
    return np.asscalar(np.abs(out))


if __name__ == "__main__":
    test = self_information(1 / 256)
    print(test)
    test = entropy(np.array([0.1, 0.5, 0.1, 0.3]))
    print(test)
    test = joint_entropy(np.array([[0.1, 0.5], [0.1, 0.3]]))
    print(test)
    test = conditional_entropy(np.array([[0.1, 0.5], [0.2, 0.3]]), np.array([0.2, 0.8]))
    print(test)
    test = mutual_information(np.array([[0.1, 0.5], [0.1, 0.3]]), np.array([0.2, 0.8]), np.array([[0.75, 0.25]]))
    print(test)

    random.seed(1)

    nd_len = 10000
    p = np.random.normal(loc=0, scale=1, size=(nd_len, ))
    q1 = np.random.normal(loc=-1, scale=1, size=(nd_len, ))
    q2 = np.random.normal(loc=1, scale=1, size=(nd_len, ))

    p = np.array(sorted(p))
    q1 = np.array(sorted(q1))
    q2 = np.array(sorted(q2))
    kl_pq1 = kl_divergence(p, q1)
    kl_pq2 = kl_divergence(p, q2)
    similar_percentage = abs(kl_pq1 - kl_pq2) / ((kl_pq1 + kl_pq2) / 2) * 100

    print(kl_pq1, kl_pq2, similar_percentage)
    kl_q2p = kl_divergence(q2, p)
    differ_percentage = abs(kl_q2p - kl_pq2) / ((kl_q2p + kl_pq2) / 2) * 100

    print(kl_q2p, differ_percentage)

    labels = np.array([0, 2])
    preds = np.array([[0.3, 0.6, 0.1], [0.2, 0.3, 0.5]])

    test = cross_entropy(preds, labels)
    print(test)