import argparse
import os

import dask
import dask.bag
import toolz as t
from dask.diagnostics import ProgressBar

import zarr

from .. import io as rio

DEFAULT_COMPRESSION = {"cname": "zstd", "clevel": 3, "shuffle": 2}


def zarrify(x, dest, chunk=512, compression=DEFAULT_COMPRESSION):
    compressor = None
    if compression:
        compressor = zarr.Blosc(**compression)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    z = zarr.open(
        dest,
        mode="w",
        shape=x.shape,
        chunks=(chunk, chunk, None),
        dtype="<u2",
        compressor=compressor)
    z[:] = x
    return z

ZARR_DEST = "{dataset}/{experiment}/Plate{plate}/{well}_s{site}.zarr"


@t.curry
def convert_to_zarr(src_base_path, dest_base_path, site_info):
    dest = os.path.join(dest_base_path, ZARR_DEST.format(**site_info))
    site_data = rio.load_site(base_path=src_base_path, **site_info)
    zarrify(site_data, dest)


def convert_all(raw_images, dest_path, metadata):
    metadata_df = rio.combine_metadata(metadata, include_controls=False)
    sites = metadata_df[['dataset', 'experiment', 'plate', 'well', 'site']].to_dict(orient='rows')
    bag = dask.bag.from_sequence(sites)
    bag.map(convert_to_zarr(raw_images, dest_path)).compute()


def cli():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Converts the raw PNGs into zarr files")
    parser.add_argument("--raw-images", type=str, help="Path of the raw images", required=True)
    parser.add_argument("--dest-path", type=str, help="Path of the zarr files to write", required=True)
    parser.add_argument("--metadata", type=str, help="Path where the metadata files live", required=True)

    args = parser.parse_args()

    from dask.diagnostics import ProgressBar
    ProgressBar().register()

    convert_all(**vars(args))



if __name__ == '__main__':
    cli()
