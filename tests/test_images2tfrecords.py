from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os
import shutil
import tempfile
import warnings
from glob import glob

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import skimage.io
import tensorflow as tf
import toolz as t

import rxrx.preprocess.images2tfrecords as sut
import rxrx.io as rio

import tfrecord_lite

FAKE_IMG_SIZE = 512
DEFAULT_SHAPE = (FAKE_IMG_SIZE, FAKE_IMG_SIZE, 6)
WELL_TYPES = ('treatment', 'negative_control', 'positive_control')
CELL_TYPES = ('HUVEC', 'U2OS', 'RPE', 'HEPG2')
DEFAULT_CHANNELS = (1, 2, 3, 4, 5, 6)
DEFAULT_SITES = (1, 2)
ALL_WELLS = [
    "%s%02d" % (chr(row), col)
    for row in range(ord('B'), ord('O') + 1) for col in range(2, 23 + 1)
]

warnings.filterwarnings('ignore', message='is a low contrast image')


def tfrecord2dicts(path, image_shape=DEFAULT_SHAPE, dtype=np.uint8):
    options = tf.python_io.TFRecordOptions(
        tf.python_io.TFRecordCompressionType.GZIP)
    it = tf.python_io.tf_record_iterator(path, options=options)
    res = []
    for e in it:
        d = tfrecord_lite.decode_example(e)
        d['image'] = np.ndarray(
            buffer=d['image'][0], dtype=dtype, shape=image_shape)
        res.append(d)
    return res


def load_tfrecords(path, image_shape=DEFAULT_SHAPE, dtype=np.uint8):
    paths = glob(os.path.join(path, '*.tfrecord'))
    return [tfrecord2dicts(p, image_shape, dtype=dtype) for p in paths]


@contextlib.contextmanager
def temp_dir(delete_dir=True, **kwargs):
    try:
        temp_dir = tempfile.mkdtemp(**kwargs)
        yield temp_dir
    finally:
        if delete_dir:
            shutil.rmtree(temp_dir)


@pytest.fixture
def tmp_dir():
    with temp_dir() as tmp:
        yield tmp


def fake_metadata(num_exps,
                  num_plates,
                  num_wells,
                  num_sites,
                  datasets=('train', 'test')):
    rows = []
    for dataset in datasets:
        for _ in range(num_exps):
            cell_type = np.random.choice(CELL_TYPES)
            experiment_number = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
            experiment = '{}-{:02d}'.format(cell_type, experiment_number)
            base_row = {
                'experiment': experiment,
                'cell_type': cell_type,
                'dataset': dataset
            }
            for p in range(1, num_plates + 1):
                base_row = t.assoc(base_row, 'plate', p)
                rand_wells = np.random.choice(
                    ALL_WELLS, num_wells, replace=False)
                for well in rand_wells:
                    well_row = t.merge(base_row, {
                        'well': well,
                        'well_type': 'treatment',
                        'sirna': 1.0
                    })
                    for site in range(1, num_sites + 1):
                        rows.append(t.assoc(well_row, 'site', site))

    df = pd.DataFrame(rows)
    return df


def create_fake_images(raw_path,
                       metadata,
                       channels=DEFAULT_CHANNELS,
                       fake_img_size=FAKE_IMG_SIZE):
    res = []
    for _, s in metadata.iterrows():
        for channel in channels:
            fname = rio.image_path(
                s['dataset'],
                s['experiment'],
                s['plate'],
                s['well'],
                s['site'],
                channel,
                base_path=raw_path)

            data = np.random.randint(
                np.iinfo(np.uint16).max, size=(fake_img_size, fake_img_size))
            data = skimage.img_as_ubyte(data)
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            skimage.io.imsave(fname, data, check_contrast=False)
            res.append(t.merge(s, {'channel': channel, 'image': data}))
    return res


def test_random_strategy_returns_determinisic_groupings():
    metadata = fake_metadata(
        datasets=('train', 'test'),
        num_exps=1,
        num_plates=1,
        num_wells=10,
        num_sites=2)

    run_1 = sut._random_partition(
        metadata_df=metadata,
        dest_path='dest',
        sites_per_tfrecord=10,
        random_seed=42)

    run_2 = sut._random_partition(
        metadata_df=metadata,
        dest_path='dest',
        sites_per_tfrecord=10,
        random_seed=42)

    assert run_1 == run_2


def test_random_strategy_is_invariant_to_datasets_being_present():
    metadata = fake_metadata(
        datasets=('train', 'test'),
        num_exps=1,
        num_plates=1,
        num_wells=10,
        num_sites=2)

    run_with_both = sut._random_partition(
        metadata_df=metadata,
        dest_path='dest',
        sites_per_tfrecord=10,
        random_seed=42)
    run_with_both = [
        group for group in run_with_both if 'train' in group['dest_path']
    ]

    train_metadata = metadata[metadata.dataset == 'train']
    run_with_just_train = sut._random_partition(
        metadata_df=train_metadata,
        dest_path='dest',
        sites_per_tfrecord=10,
        random_seed=42)

    assert run_with_both == run_with_just_train


def test_by_exp_plate_site_strategy_returns_determinisic_groupings():
    metadata = fake_metadata(
        datasets=('train', 'test'),
        num_exps=1,
        num_plates=1,
        num_wells=10,
        num_sites=2)

    run_1 = sut._by_exp_plate_site(
        metadata_df=metadata, dest_path='dest', random_seed=42)

    run_2 = sut._by_exp_plate_site(
        metadata_df=metadata, dest_path='dest', random_seed=42)

    assert run_1 == run_2


def test_by_exp_plate_site_strategy_is_invariant_to_datasets_being_present():
    metadata = fake_metadata(
        datasets=('train', 'test'),
        num_exps=1,
        num_plates=1,
        num_wells=10,
        num_sites=2)

    run_with_both = sut._by_exp_plate_site(
        metadata_df=metadata, dest_path='dest', random_seed=42)

    train_metadata = metadata[metadata.dataset == 'train']
    run_with_just_train = sut._by_exp_plate_site(
        metadata_df=train_metadata, dest_path='dest', random_seed=42)
    training_paths = set([p['dest_path'] for p in run_with_just_train])
    run_with_both = [
        v for v in run_with_both if v['dest_path'] in training_paths
    ]

    assert run_with_both == run_with_just_train


def test_random_tfrecord_packing(tmp_dir):
    num_wells = 6
    num_sites = 1
    num_site_images = num_wells * num_sites
    metadata = fake_metadata(
        datasets=('train', 'test'),
        num_exps=1,
        num_plates=1,
        num_wells=num_wells,
        num_sites=num_sites)
    raw_path = os.path.join(tmp_dir, 'raw')
    channels_to_create = [1, 2]
    num_channels = len(channels_to_create)
    fake_sites = create_fake_images(raw_path, metadata, channels_to_create)
    dest_path = os.path.join(tmp_dir, 'tfrecords')
    seed = 42
    sites_per_tfrecord = num_site_images // 2
    expected_num_tfrecords = num_site_images // sites_per_tfrecord
    sut.pack_tfrecords(
        raw_path,
        metadata,
        num_workers=4,
        dest_path=dest_path,
        strategies=['random'],
        random_seeds=[seed],
        channels=channels_to_create,
        sites_per_tfrecord=sites_per_tfrecord)

    train_dest = os.path.join(dest_path, "random-{}".format(seed), 'train')
    test_dest = os.path.join(dest_path, "random-{}".format(seed), 'test')
    image_shape = (FAKE_IMG_SIZE, FAKE_IMG_SIZE, num_channels)

    train_results = load_tfrecords(train_dest, image_shape)
    expected_counts = [sites_per_tfrecord] * expected_num_tfrecords
    assert [len(r) for r in train_results] == expected_counts

    test_results = load_tfrecords(test_dest, image_shape)
    assert [len(r) for r in test_results] == expected_counts

    ## basic smoke test to make sure keys are being written
    dict_example = train_results[0][0]
    assert set(dict_example.keys()) == {
        'image', 'well_type', 'experiment', 'site', 'well', 'sirna', 'plate',
        'cell_type'
    }

    assert dict_example['image'].shape == (512, 512, 2)


def test_packing_by_exp_plate_site(tmp_dir):
    num_wells = 6
    num_sites = 1
    num_site_images = num_wells * num_sites
    metadata = fake_metadata(
        datasets=('train', 'test'),
        num_exps=1,
        num_plates=1,
        num_wells=num_wells,
        num_sites=num_sites)
    raw_path = os.path.join(tmp_dir, 'raw')
    channels_to_create = [1, 2]
    num_channels = len(channels_to_create)
    fake_sites = create_fake_images(raw_path, metadata, channels_to_create)
    dest_path = os.path.join(tmp_dir, 'tfrecords')
    seed = 42
    sites_per_tfrecord = num_site_images // 2
    image_shape = (FAKE_IMG_SIZE, FAKE_IMG_SIZE, num_channels)
    sut.pack_tfrecords(
        raw_path,
        metadata,
        num_workers=4,
        dest_path=dest_path,
        strategies=['by_exp_plate_site'],
        random_seeds=[seed],
        channels=channels_to_create,
        sites_per_tfrecord=sites_per_tfrecord)

    expected_tfrecords = set([
        'by_exp_plate_site-42/{}_p{}_s{}.tfrecord'.format(
            row["experiment"], row["plate"], row["site"])
        for _, row in metadata.iterrows()
    ])
    expected_files = set(
        [os.path.join(dest_path, f) for f in expected_tfrecords])
    files = set(
        glob(os.path.join(dest_path, "by_exp_plate_site-42/*.tfrecord")))

    assert files == expected_files
