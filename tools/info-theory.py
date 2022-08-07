import numpy as np


def self_information(p):
    return -np.log2(p)


def entropy(p):
    entropy = - p * np.log2(p)
    # Operator `nansum` will sum up the non-nan number
    out = np.nansum(entropy)
    return out


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