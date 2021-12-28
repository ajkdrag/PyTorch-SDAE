from torchvision import transforms


def get_transform(img_size):
    new_h, new_w = img_size
    return transforms.Compose(
        [transforms.Resize(size=(new_h, new_w)), transforms.ToTensor()]
    )


def get_inv_transform():
    # no op as we didn't do any transforms which need to be "inverted"
    return lambda x: x

