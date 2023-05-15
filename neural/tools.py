from PIL import Image
import numpy as np
import skimage
from skimage.transform import rescale, resize, downscale_local_mean


def load_image(img_path):
    img = Image.open(img_path)

    img.load()
    img_array = np.asarray(img, dtype='int32')

    return img_array


def image_downsample(img_array):
    ds_array = img_array / 255

    d1 = ds_array.shape[0] // 28 + 1
    d2 = ds_array.shape[1] // 28 + 1

    resized_image = resize(ds_array, (28, 28),
                           anti_aliasing=True)

    r = skimage.measure.block_reduce(ds_array[:, :, 0],
                                     (d1, d2),
                                     np.mean)
    g = skimage.measure.block_reduce(ds_array[:, :, 1],
                                     (d1, d2),
                                     np.mean)
    b = skimage.measure.block_reduce(ds_array[:, :, 2],
                                     (d1, d2),
                                     np.mean)

    ds_array = np.stack((r, g, b), axis=-1)

    return resized_image


def convert_image_to_grayscale(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])
