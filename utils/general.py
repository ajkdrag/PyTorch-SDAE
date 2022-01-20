import torch
import torch.nn.functional as F


def get_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def get_children(model: torch.nn.Module):
    children = list(model.children())
    flatt_children = []
    if children == []:
        return model
    else:
        for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children


def kl_divergence(p, q):
    # bernoulli dist
    s1 = torch.sum(p * torch.log(p / q))
    s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
    return s1 + s2
