# copyright: https://github.com/ildoonet/pytorch-randaugment
# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
# This code is modified version of one of ildoonet, for randaugmentation of fixmatch.

import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from PIL import Image


def AutoContrast(img, _):
    """
    Convert an image to an image.

    Args:
        img: (array): write your description
        _: (todo): write your description
    """
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v):
    """
    Convert an image to an image.

    Args:
        img: (array): write your description
        v: (todo): write your description
    """
    assert v >= 0.0
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v):
    """
    Convert an image to an image.

    Args:
        img: (array): write your description
        v: (str): write your description
    """
    assert v >= 0.0
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v):
    """
    Convert an image in antsimage.

    Args:
        img: (array): write your description
        v: (todo): write your description
    """
    assert v >= 0.0
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Equalize(img, _):
    """
    Equalize an image.

    Args:
        img: (array): write your description
        _: (array): write your description
    """
    return PIL.ImageOps.equalize(img)


def Invert(img, _):
    """
    Convert an image to an image.

    Args:
        img: (array): write your description
        _: (todo): write your description
    """
    return PIL.ImageOps.invert(img)


def Identity(img, v):
    """
    Return the identity of an image

    Args:
        img: (array): write your description
        v: (array): write your description
    """
    return img


def Posterize(img, v):  # [4, 8]
    """
    Convert an image to poster.

    Args:
        img: (array): write your description
        v: (todo): write your description
    """
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v):  # [-30, 30]
    """
    Rotate an image

    Args:
        img: (array): write your description
        v: (int): write your description
    """
    #assert -30 <= v <= 30
    #if random.random() > 0.5:
    #    v = -v
    return img.rotate(v)



def Sharpness(img, v):  # [0.1,1.9]
    """
    Convert an image to an image.

    Args:
        img: (array): write your description
        v: (todo): write your description
    """
    assert v >= 0.0
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v):  # [-0.3, 0.3]
    """
    Transform an image

    Args:
        img: (array): write your description
        v: (array): write your description
    """
    #assert -0.3 <= v <= 0.3
    #if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    """
    Transform an image

    Args:
        img: (array): write your description
        v: (array): write your description
    """
    #assert -0.3 <= v <= 0.3
    #if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    """
    Convert an image to antsimage

    Args:
        img: (array): write your description
        v: (array): write your description
    """
    #assert -0.3 <= v <= 0.3
    #if random.random() > 0.5:
    #    v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    """
    Convert image to image

    Args:
        img: (array): write your description
        v: (todo): write your description
    """
    #assert v >= 0.0
    #if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    """
    Convert antsimage

    Args:
        img: (array): write your description
        v: (array): write your description
    """
    #assert -0.3 <= v <= 0.3
    #if random.random() > 0.5:
    #    v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    """
    Convert an image to an image

    Args:
        img: (array): write your description
        v: (todo): write your description
    """
    #assert 0 <= v
    #if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Solarize(img, v):  # [0, 256]
    """
    Convert an rgb image.

    Args:
        img: (array): write your description
        v: (array): write your description
    """
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def Cutout(img, v):  #[0, 60] => percentage: [0, 0.2] => change to [0, 0.5]
    """
    Convert an image to an image

    Args:
        img: (array): write your description
        v: (array): write your description
    """
    assert 0.0 <= v <= 0.5
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    """
    Draws an image with the image

    Args:
        img: (array): write your description
        v: (int): write your description
    """
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img

    
def augment_list():  
    """
    Create a list.

    Args:
    """
    l = [
        (AutoContrast, 0, 1),
        (Brightness, 0.05, 0.95),
        (Color, 0.05, 0.95),
        (Contrast, 0.05, 0.95),
        (Equalize, 0, 1),
        (Identity, 0, 1),
        (Posterize, 4, 8),
        (Rotate, -30, 30),
        (Sharpness, 0.05, 0.95),
        (ShearX, -0.3, 0.3),
        (ShearY, -0.3, 0.3),
        (Solarize, 0, 256),
        (TranslateX, -0.3, 0.3),
        (TranslateY, -0.3, 0.3)
    ]
    return l

    
class RandAugment:
    def __init__(self, n, m):
        """
        Initialize the list.

        Args:
            self: (todo): write your description
            n: (int): write your description
            m: (int): write your description
        """
        self.n = n
        self.m = m      # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list()

        
    def __call__(self, img):
        """
        Return a random variate method

        Args:
            self: (todo): write your description
            img: (todo): write your description
        """
        ops = random.choices(self.augment_list, k=self.n)
        for op, min_val, max_val in ops:
            val = min_val + float(max_val - min_val)*random.random()
            img = op(img, val) 
        cutout_val = random.random() * 0.5 
        img = Cutout(img, cutout_val) #for fixmatch
        return img

    
if __name__ == '__main__':
    randaug = RandAugment(3,5)
    print(randaug)
    for item in randaug.augment_list:
        print(item)
    