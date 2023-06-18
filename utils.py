import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def print_img_w_ellipsis(N, img, zt, mu_ht, var_ht, theta_ht, eigen, gray_color=True):
    fig = plt.figure()
    ax = fig.add_subplot()
    color = "lightgray" if gray_color else "red"
    img_np = np.transpose(img.detach().cpu().numpy(), (1, 2, 0))
    grip = zt.detach().cpu().numpy() * N
    obj = mu_ht.detach().cpu().numpy() * N

    ax.imshow(img_np)

    ax.scatter([obj[1], grip[1]], [obj[0], grip[0]], color=color, s=80)

    angle = theta_ht[0] * 360 / (2 * np.pi)
    eigen2 = 0.15 * N if eigen[0] != eigen[1] else 0.2 * N
    eigen1 = 0.4 * N if eigen[0] != eigen[1] else eigen2
    ellipse = Ellipse(xy=(0, 0), width=eigen1, height=eigen2, angle=angle, edgecolor=color, linewidth=3)  # eigen[0] eigen[1]
    transf = transforms.Affine2D().translate(obj[1], obj[0])
    ellipse.set_transform(transf + ax.transData)
    ellipse.set_facecolor("none")
    ax.add_patch(ellipse)

    return fig


def print_img_w_points(N, img, zt, ht, gray_color=True):
    fig = plt.figure()
    ax = fig.add_subplot()
    color = "lightgray" if gray_color else "red"
    img_np = np.transpose(img.detach().cpu().numpy(), (1, 2, 0))
    grip = zt.detach().cpu().numpy() * N
    obj = ht.detach().cpu().numpy() * N

    ax.imshow(img_np)
    ax.scatter([obj[1], grip[1]], [obj[0], grip[0]], color=color, s=80)

    return fig


def spatial_softmax(features):
    """Compute softmax over the spatial dimensions
    Compute the softmax over heights and width
    Args
    ----
    features: tensor of shape [N, C, H, W]
    """
    features_reshape = features.reshape(features.shape[:-2] + (-1,))
    output = F.softmax(features_reshape, dim=-1)
    output = output.reshape(features.shape)
    return output


def _maybe_convert_dict(value):
    if isinstance(value, dict):
        return ConfigDict(value)

    return value


class ConfigDict(dict):
    """Configuration container class."""

    def __init__(self, initial_dictionary=None):
        """Creates an instance of ConfigDict.
        Args:
            initial_dictionary: Optional dictionary or ConfigDict containing initial
            parameters.
        """
        if initial_dictionary:
            for field, value in initial_dictionary.items():
                initial_dictionary[field] = _maybe_convert_dict(value)
        super(ConfigDict, self).__init__(initial_dictionary)

    def __setattr__(self, attribute, value):
        self[attribute] = _maybe_convert_dict(value)

    def __getattr__(self, attribute):
        try:
            return self[attribute]
        except KeyError as e:
            raise AttributeError(e)

    def __delattr__(self, attribute):
        try:
            del self[attribute]
        except KeyError as e:
            raise AttributeError(e)

    def __setitem__(self, key, value):
        super(ConfigDict, self).__setitem__(key, _maybe_convert_dict(value))


def get_random_color(pastel_factor=0.5):
    return [(x + pastel_factor) / (1.0 + pastel_factor)
            for x in [np.random.uniform(0, 1.0) for i in [1, 2, 3]]]


def color_distance(c1, c2):
    return sum([abs(x[0] - x[1]) for x in zip(c1, c2)])


def generate_new_color(existing_colors, pastel_factor=0.5):
    max_distance = None
    best_color = None
    for i in range(0, 100):
        color = get_random_color(pastel_factor=pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color, c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color


def get_n_colors(n, pastel_factor=0.9):
    colors = []
    for i in range(n):
        colors.append(generate_new_color(colors, pastel_factor=0.9))
    return colors


def compute_otsu_criteria(im, th):
    # create the thresholded image
    thresholded_im = np.zeros(im.shape)
    thresholded_im[im >= th] = 1

    # compute weights
    nb_pixels = im.size
    nb_pixels1 = np.count_nonzero(thresholded_im)
    weight1 = nb_pixels1 / nb_pixels
    weight0 = 1 - weight1

    # if one the classes is empty, eg all pixels are below or above the threshold, that threshold will not be considered
    # in the search for the best threshold
    if weight1 == 0 or weight0 == 0:
        return np.inf

    # find all pixels belonging to each class
    val_pixels1 = im[thresholded_im == 1]
    val_pixels0 = im[thresholded_im == 0]

    # compute variance of these classes
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0
    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0

    return weight0 * var0 + weight1 * var1


def otsu(arr):
    criterias = [compute_otsu_criteria(arr, th) for th in arr]
    return np.argmin(criterias)