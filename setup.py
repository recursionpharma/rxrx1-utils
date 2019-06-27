# this setup file is for the dataflow job only at this point
import setuptools

# Configure the required packages and scripts to install.
# Note that the Python Dataflow containers come with numpy already installed
# so this dependency will not trigger anything to be installed unless a version
# restriction is specified.
REQUIRED_PACKAGES = [
    'toolz scikit-image pandas dask'
    ' pytest'
    ' gcsfs tensorflow'.split()
]

setuptools.setup(
    name='rxrx1-utils',
    version='0.0.1',
    description='Example pipeline to pack raw tifs from rxrx1 dataset into TFRecord',
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages(),)
