import argparse
import datetime
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import tensorflow as tf
import toolz as t

from .. import io as rio
from .. import utils as rutils
from ..io import DEFAULT_CHANNELS

TFRECORD_COMPRESSION = tf.python_io.TFRecordCompressionType.GZIP
TFRECORD_OPTIONS = tf.python_io.TFRecordOptions(TFRECORD_COMPRESSION)
VALID_DATASETS = {'train', 'test'}
VALID_STRATEGIES = {'random', 'by_exp_plate_site'}

### TensorFlow Helpers


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def string_feature(value):
    return bytes_feature(value.encode('utf-8'))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(
        value=rutils.wrap(value)))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(
        value=ruitls.wrap(value)))


### Conversion to TFExample and TFRecord logic


def dict_to_tfexample(site):
    """
    Takes a dictionary of a site with all the metadata and the `image` data.

    Returns a TFExample
    """

    features = {
        'image': bytes_feature(site['image'].tostring()),
        'well': string_feature(site['well']),
        'well_type': string_feature(site['well_type']),
        'experiment': string_feature(site['experiment']),
        'plate': int64_feature(site['plate']),
        'site': int64_feature(site['site']),
        'cell_type': string_feature(site['cell_type'])
    }

    # Handle case where sirna is not known (test)
    if site["sirna"] is not None:
        features["sirna"] = int64_feature(site["sirna"])

    return tf.train.Example(features=tf.train.Features(feature=features))


def _pack_tfrecord(base_path,
                   sites,
                   dest_path,
                   channels=DEFAULT_CHANNELS):
    if not dest_path.startswith('gs://'):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with tf.python_io.TFRecordWriter(
            dest_path, options=TFRECORD_OPTIONS) as writer:
        for site in sites:
            data = rio.load_site(
                base_path=base_path,
                channels=channels,
                **rutils.select_keys(site, ('dataset', 'experiment', 'plate',
                                            'well', 'site')))
            example = dict_to_tfexample(t.assoc(site, 'image', data))
            writer.write(example.SerializeToString())


### Strategies to pack the TFRecords differently and some helper functions
#
# Each strategy takes the metadata DataFrame and returns a list of
# dictionaries containing `dest_path` and `sites` where
# `dest_path` - the full path of the destination TFRecord file
# `sites` - a list of all of the sites that should be packed into the
#           destination path. Each `site` is a row, in dictionary form,
#           from the metadata dataframe.
#


def _dataset_rs_dict(seed):
    """Returns a dictionary of random states keyed by dataset.
    A seed for every dataset is created regardless of if it will be
    processed. This is done to guarantee determinism of the
    randomization invariant of what datasets are being processed.
    """
    rs = np.random.RandomState(seed)
    high = 2**32 - 1
    return {
        ds: np.random.RandomState(rs.randint(high))
        for ds in sorted(VALID_DATASETS)
    }


def _correct_sirna_dtype(row):
    if np.isnan(row['sirna']):
        row['sirna'] = None
    else:
        row['sirna'] = int(row['sirna'])
    return row


def _random_partition(metadata_df,
                      dest_path,
                      sites_per_tfrecord=308,
                      random_seed=42):
    """
    Randomly partitions each dataset into multiple TFRecords.
    """
    # make groupby's determinisic
    metadata_df = metadata_df.sort_values(
        ['dataset', 'experiment', 'plate', 'well', 'site'])
    # get random states to make randomizations determinisic
    rs_dict = _dataset_rs_dict(random_seed)

    to_pack = []
    for dataset, df in metadata_df.groupby('dataset'):
        df = (df.sort_values(['experiment', 'plate', 'well', 'site'])
              .sample(frac=1.0, random_state=rs_dict[dataset]))
        rows = [_correct_sirna_dtype(row) for row in df.to_dict(orient='row')]
        sites_for_files = t.partition_all(sites_per_tfrecord, rows)
        dataset_path = os.path.join(dest_path, 'random-{}'.format(random_seed), dataset)
        for file_num, sites in enumerate(sites_for_files, 1):
            dest_file = os.path.join(dataset_path, "{:03d}.tfrecord".format(file_num))
            to_pack.append({'dest_path': dest_file, 'sites': sites})
    return to_pack


def _by_exp_plate_site(metadata_df, dest_path, random_seed=42):
    """
    Groups by experiment, plate, and packs each site into individual TFRecords.
    """
    # make groupby's determinisic
    metadata_df = metadata_df.sort_values(
        ['dataset', 'experiment', 'plate', 'well', 'site'])
    # get random states to make randomizations determinisic
    rs_dict = _dataset_rs_dict(random_seed)

    to_pack = []
    for (dataset, exp, plate, site), df in metadata_df.groupby(
        ['dataset', 'experiment', 'plate', 'site']):
        df = (df.sort_values(['experiment', 'plate', 'well', 'site'])
              .sample(frac=1.0, random_state=rs_dict[dataset]))
        rows = [_correct_sirna_dtype(row) for row in df.to_dict(orient='row')]

        dest_file = os.path.join(dest_path, 'by_exp_plate_site-{}'.format(random_seed),
                                 "{}_p{}_s{}.tfrecord".format(exp, plate, site))
        to_pack.append({'dest_path': dest_file, 'sites': rows})
    return to_pack


def _sites_df(i, ix):
    return pd.DataFrame([i] * len(ix), index=ix, columns=['site'])


### Main entry point and CLI logic


def pack_tfrecords(images_path,
                   metadata_df,
                   num_workers,
                   dest_path,
                   strategies=['random', 'by_exp_plate_site'],
                   channels=DEFAULT_CHANNELS,
                   sites_per_tfrecord=308,
                   random_seeds=[42],
                   runner='dask',
                   project=None,
                   datasets=None):
    if datasets is None:
        datasets = [
            ds.strip('/') for ds in tf.gfile.ListDirectory(images_path)
            if ds.strip('/') in VALID_DATASETS
        ]

    # Only consider metadata for the datasets we care about
    metadata_df = metadata_df[metadata_df.dataset.isin(datasets)]
    # only pack images for the treatment wells, not the controls!
    metadata_df = metadata_df[metadata_df.well_type == "treatment"]

    strategies = set(strategies)

    if len(strategies - VALID_STRATEGIES) > 0:
        raise ValueError(
            'invalid strategies: {}. You may only provide a subset of {}'.format(strategies, VALID_STRATEGIES)
        )

    to_pack = []
    for random_seed in random_seeds:
        if 'random' in strategies:
            to_pack += _random_partition(
                metadata_df,
                dest_path,
                random_seed=random_seed,
                sites_per_tfrecord=sites_per_tfrecord)

        if 'by_exp_plate_site' in strategies:
            to_pack += _by_exp_plate_site(
                metadata_df, dest_path, random_seed=random_seed)

    if runner == 'dask':
        import dask
        import dask.bag

        print('Distributing {} on dask'.format(len(to_pack)))
        to_pack_bag = dask.bag.from_sequence(to_pack, npartitions=len(to_pack))
        (to_pack_bag
         .map(lambda kw: _pack_tfrecord(base_path=images_path,
                                        channels=channels,
                                        **kw))
         .compute(num_workers=num_workers))
        return [p['dest_path'] for p in to_pack]
    else:
        print('Distributing {} on {}'.format(len(to_pack), runner))
        run_on_dataflow(to_pack, dest_path, images_path, channels, runner, project)
        return None


def run_on_dataflow(to_pack, dest_path, images_path, channels, runner, project):

    import apache_beam as beam

    options = {
        'staging_location':
        os.path.join(dest_path, 'tmp', 'staging'),
        'temp_location':
        os.path.join(dest_path, 'tmp'),
        'job_name': ('rxrx1-' + os.getlogin().replace('.', '-') + '-' +
                     datetime.datetime.now().strftime('%y%m%d-%H%M%S')),
        'max_num_workers':
        600,  # CHANGE AS NEEDED
        'machine_type':
        'n1-standard-4',
        'save_main_session':
        True,
        'setup_file': (os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '../../setup.py')),
        'runner':
        runner,
        'project':
        project
    }
    opts = beam.pipeline.PipelineOptions(flags=[], **options)
    with beam.Pipeline(runner, options=opts) as p:
        (p
                | 'find_images' >> beam.Create(to_pack)
                | 'pack' >> beam.FlatMap(
                    lambda kw: _pack_tfrecord(base_path=images_path,
                                              channels=channels,
                                              **kw))
        )
        if runner == 'dataflow':
            print(
                'Submitting job ... Please monitor at https://console.cloud.google.com/dataflow'
            )


def cli():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Packs the raw PNG images into TFRecords.")
    parser.add_argument("--raw-images", type=str, help="Path of the raw images",
                        default=rio.DEFAULT_IMAGES_BASE_PATH)
    parser.add_argument(
        "--metadata", type=str, help="Path to the metadata directory",
        default=rio.DEFAULT_METADATA_BASE_PATH)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of workers to be writing TFRecords. Defaults to number of cores."
    )
    parser.add_argument(
        "--random-seeds",
        type=int,
        nargs='+',
        default=[42],
        help="The seed used to make the sorting determistic. Embedded in the dir name to allow multiple folds to be created."
    )
    parser.add_argument(
        "--sites-per-tfrecord",
        type=int,
        default=1500,
        help="Only used with the random strategy, indicates how many site images you want in a single TFRecord"
    )
    parser.add_argument(
        "--strategies",
        nargs='+',
        choices=VALID_STRATEGIES,
        default=['random', 'by_exp_plate_site'],
        help="""What strategies to use to pack up the records:
\t`random` - Randomly partitions each dataset into multiple TFRecords.
\t`by_exp_plate_site` - Groups by experiment, plate, and packs each site into individual TFRecords.
                        """)
    parser.add_argument(
        "--dest-path",
        type=str,
        default="./tfrecords",
        help="Destination directory of where to write the tfrecords")
    parser.add_argument(
        "--runner",
        type=str,
        default="dask",
        choices={'dask', 'dataflow'},
        help="Specify one of DirectRunner, dataflow, or dask")
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="If using dataflow, the project to bill")
    args = parser.parse_args()
    if args.runner == 'dataflow':
        if not args.project:
            raise ValueError('When using dataflow, you need to specify project')

    metadata_df = rio.combine_metadata(args.metadata)
    if args.runner == 'dask':
        from dask.diagnostics import ProgressBar
        ProgressBar().register()

    pack_tfrecords(
        images_path=args.raw_images,
        metadata_df=metadata_df,
        dest_path=args.dest_path,
        strategies=args.strategies,
        sites_per_tfrecord=args.sites_per_tfrecord,
        random_seeds=args.random_seeds,
        num_workers=args.num_workers,
        runner=args.runner,
        project=args.project)


if __name__ == '__main__':
    cli()
