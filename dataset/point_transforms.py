
import numpy as np
from skimage import color


class RandomShear:
    def __call__(self, xyz):
        matrix = np.random.randn(3, 3)
        T = np.eye(3) + 0.1 * matrix
        return xyz @ T


class RandomTranslation:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, xyz):
        factors = np.random.randn(1, 3)*[0, 1, 0]  # only stature direction
        trans = self.scale * factors
        return xyz+trans


class RandomGaussianNoise:
    def __init__(self, mean=0, var=0.001):
        self.mean = mean
        self.var = var

    def __call__(self, colors):
        noise = np.random.normal(self.mean, self.var ** 0.5, colors.shape)

        return colors+noise


class RandomValue:
    def __init__(self, min=-0.2, max=0.2):
        self.scale = max-min
        self.bias = min

    def __call__(self, colors):

        offset = np.random.rand()
        colors_hsv = color.rgb2hsv(colors)  # transform colors to hsv space
        colors_hsv[..., -1] += self.scale * offset + self.bias  # apply augmentation
        colors_rgb = color.hsv2rgb(colors_hsv)  # transform colors back to rgb space

        return colors_rgb


class RandomSaturation:
    def __init__(self, min=-0.15, max=0.15):
        self.scale = max-min
        self.bias = min

    def __call__(self, colors):

        offset = np.random.rand()
        colors_hsv = color.rgb2hsv(colors)  # transform colors to hsv space
        colors_hsv[:, 1] += self.scale * offset + self.bias  # apply augmentation
        colors_rgb = color.hsv2rgb(colors_hsv)  # transform colors back to rgb space
        return colors_rgb
