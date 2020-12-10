import matplotlib.pyplot as plt


def plot_img_and_target(img, target):
    classes = target.shape[2] if len(target.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i+1].set_title(f'Output target (class {i+1})')
            ax[i+1].imshow(target[:, :, i])
    else:
        ax[1].set_title(f'Output target')
        ax[1].imshow(target)
    plt.xticks([]), plt.yticks([])
    plt.show()
