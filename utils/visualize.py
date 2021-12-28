import matplotlib.pyplot as plt


def plot_one_list(list_images, rows=4, cols=4, out=None):
    plt.figure(figsize=(cols * 2, rows * 2))
    total = rows * cols
    for idx in range(total):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(list_images[idx])
    if out:
        plt.savefig(out)


def plot_two_lists(list_images_1, list_images_2, rows=4, cols=4, out=None):
    plt.figure(figsize=(cols * 2, rows * 2))
    list_1_idx = 0
    list_2_idx = 0
    for idx in range(cols * rows):
        plt.subplot(rows, cols, idx + 1)
        row_id = idx // cols
        if row_id & 1:
            plt.imshow(list_images_2[list_2_idx].squeeze())
            list_2_idx += 1
        else:
            plt.imshow(list_images_1[list_1_idx].squeeze())
            list_1_idx += 1
    if out:
        plt.savefig(out)
