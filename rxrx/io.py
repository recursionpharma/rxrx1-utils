from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from skimage.io import imread
import pandas as pd

import tensorflow as tf

DEFAULT_BASE_PATH = 'gs://rxrx1-us-central1'
DEFAULT_METADATA_BASE_PATH = os.path.join(DEFAULT_BASE_PATH, 'metadata')
DEFAULT_IMAGES_BASE_PATH = os.path.join(DEFAULT_BASE_PATH, 'images')
DEFAULT_CHANNELS = (1, 2, 3, 4, 5, 6)
RGB_MAP = {
    1: {
        'rgb': np.array([19, 0, 249]),
        'range': [0, 51]
    },
    2: {
        'rgb': np.array([42, 255, 31]),
        'range': [0, 107]
    },
    3: {
        'rgb': np.array([255, 0, 25]),
        'range': [0, 64]
    },
    4: {
        'rgb': np.array([45, 255, 252]),
        'range': [0, 191]
    },
    5: {
        'rgb': np.array([250, 0, 253]),
        'range': [0, 89]
    },
    6: {
        'rgb': np.array([254, 255, 40]),
        'range': [0, 191]
    }
}


def load_image(image_path):
    with tf.io.gfile.GFile(image_path, 'rb') as f:
        return imread(f, format='png')


def load_images_as_tensor(image_paths, dtype=np.uint8):
    n_channels = len(image_paths)

    data = np.ndarray(shape=(512, 512, n_channels), dtype=dtype)

    for ix, img_path in enumerate(image_paths):
        data[:, :, ix] = load_image(img_path)

    return data


def convert_tensor_to_rgb(t, channels=DEFAULT_CHANNELS, vmax=255, rgb_map=RGB_MAP):
    """
    Converts and returns the image data as RGB image

    Parameters
    ----------
    t : np.ndarray
        original image data
    channels : list of int
        channels to include
    vmax : int
        the max value used for scaling
    rgb_map : dict
        the color mapping for each channel
        See rxrx.io.RGB_MAP to see what the defaults are.

    Returns
    -------
    np.ndarray the image data of the site as RGB channels
    """
    colored_channels = []
    for i, channel in enumerate(channels):
        x = (t[:, :, i] / vmax) / \
            ((rgb_map[channel]['range'][1] - rgb_map[channel]['range'][0]) / 255) + \
            rgb_map[channel]['range'][0] / 255
        x = np.where(x > 1., 1., x)
        x_rgb = np.array(
            np.outer(x, rgb_map[channel]['rgb']).reshape(512, 512, 3),
            dtype=int)
        colored_channels.append(x_rgb)
    im = np.array(np.array(colored_channels).sum(axis=0), dtype=int)
    im = np.where(im > 255, 255, im)
    return im


def image_path(dataset,
               experiment,
               plate,
               address,
               site,
               channel,
               base_path=DEFAULT_IMAGES_BASE_PATH):
    """
    Returns the path of a channel image.

    Parameters
    ----------
    dataset : str
        what subset of the data: train, test
    experiment : str
        experiment name
    plate : int
        plate number
    address : str
        plate address
    site : int
        site number
    channel : int
        channel number
    base_path : str
        the base path of the raw images

    Returns
    -------
    str the path of image
    """
    return os.path.join(base_path, dataset, experiment, "Plate{}".format(plate),
                        "{}_s{}_w{}.png".format(address, site, channel))


def load_site(dataset,
              experiment,
              plate,
              well,
              site,
              channels=DEFAULT_CHANNELS,
              base_path=DEFAULT_IMAGES_BASE_PATH):
    """
    Returns the image data of a site

    Parameters
    ----------
    dataset : str
        what subset of the data: train, test
    experiment : str
        experiment name
    plate : int
        plate number
    address : str
        plate address
    site : int
        site number
    channels : list of int
        channels to include
    base_path : str
        the base path of the raw images

    Returns
    -------
    np.ndarray the image data of the site
    """
    channel_paths = [
        image_path(
            dataset, experiment, plate, well, site, c, base_path=base_path)
        for c in channels
    ]
    return load_images_as_tensor(channel_paths)


def load_site_as_rgb(dataset,
                     experiment,
                     plate,
                     well,
                     site,
                     channels=DEFAULT_CHANNELS,
                     base_path=DEFAULT_IMAGES_BASE_PATH,
                     rgb_map=RGB_MAP):
    """
    Loads and returns the image data as RGB image

    Parameters
    ----------
    dataset : str
        what subset of the data: train, test
    experiment : str
        experiment name
    plate : int
        plate number
    address : str
        plate address
    site : int
        site number
    channels : list of int
        channels to include
    base_path : str
        the base path of the raw images
    rgb_map : dict
        the color mapping for each channel
        See rxrx.io.RGB_MAP to see what the defaults are.

    Returns
    -------
    np.ndarray the image data of the site as RGB channels
    """
    x = load_site(dataset, experiment, plate, well, site, channels, base_path)
    return convert_tensor_to_rgb(x, channels, rgb_map=rgb_map)


def _tf_read_csv(path):
    with tf.io.gfile.GFile(path, 'rb') as f:
        return pd.read_csv(f)


def _load_dataset(base_path, dataset, include_controls=True):
    df = _tf_read_csv(os.path.join(base_path, dataset + '.csv'))
    if include_controls:
        controls = _tf_read_csv(
            os.path.join(base_path, dataset + '_controls.csv'))
        df['well_type'] = 'treatment'
        df = pd.concat([controls, df], sort=True)
    df['cell_type'] = df.experiment.str.split("-").apply(lambda a: a[0])
    df['dataset'] = dataset
    dfs = []
    for site in (1, 2):
        df = df.copy()
        df['site'] = site
        dfs.append(df)
    res = pd.concat(dfs).sort_values(
        by=['id_code', 'site']).set_index('id_code')
    return res


def combine_metadata(base_path=DEFAULT_METADATA_BASE_PATH,
                     include_controls=True):
    """
    Combines all metadata files into a single dataframe and
    expands it to include sites, not just wells.

    Note, that the dtype of sirna is a float due to the missing
    test values but it should be treated as an int.

    Parameters
    ----------
    base_path : str
        where the metadata files from Kaggle live
    include_controls : bool
        indicate if you want the controls included in the dataframe

    Returns
    -------
    pandas.DataFrame the combined metadata
    """
    df = pd.concat(
        [
            _load_dataset(
                base_path, dataset, include_controls=include_controls)
            for dataset in ['test', 'train']
        ],
        sort=True)
    return df
