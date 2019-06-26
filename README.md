# rxrx1-utils

Starter code for the CellSignal NeurIPS 2019 competition [hosted on Kaggle](https://www.kaggle.com/c/recursion-cellular-image-classification).

To learn more about the dataset please visit [RxRx.ai](http://rxrx.ai).

## Notebooks

Here are some notebooks to illustrate how this code can be used.

 * [Image visualization][vis-notebook]
 * [Model training on TPUs][training-notebook]
 
 [vis-notebook]: https://colab.research.google.com/github/recursionpharma/rxrx1-utils/blob/master/notebooks/visualization.ipynb
 [training-notebook]: https://colab.research.google.com/github/recursionpharma/rxrx1-utils/blob/master/notebooks/training.ipynb
 
## Setup

This starter code works with python 2.7 and above. To install the deps needed for training and visualization run:

```
pip install -r  requirements.txt
```

If you plan on using the preprocessing functionality you also need to install other deps:

```
pip install -r preprocessing_requirements.txt
```

## Preprocessing

### images2tfrecords

Script that packs raw images from the `rxrx1` dataset into `TFRecord`s. This scripts runs locally or using Google DataFlow.

Run `python -m rxrx.preprocess.images2tfrecords --help` for usage instructions.

## Training on TPUs

This repo has barebones starter code on how to train a model on the RxRx1 dataset using Google Cloud TPUs.

The easiest way to see this in action is to look at this [notebook][training-notebook].

You can also spin up a VM to launch jobs from. To understand TPUs the best place to start is the [TPU quickstart guide][tpu-quickstart]. The `ctpu` command is helpful and you can find [its documentation][ctpu-docs] here. Note, that you can easily [download and install ctpu][download-ctpu] to you local machine.

[tpu-quickstart]: https://cloud.google.com/tpu/docs/quickstart
[ctpu-docs]: https://cloud.google.com/tpu/docs/ctpu-reference
[download-ctpu]: https://github.com/tensorflow/tpu/tree/master/tools/ctpu#download


TODO: finish this part

Below is an example workflow:

1. Spin up a VM to using `ctpu`
1. ssh into the tpu vm
1. Setup the repo and install the dependencies
```
git clone git@github.com:recursionpharma/rxrx1-utils.git
cd rxrx1-utils
pip install -r requirements.txt
```
1. export TPU_NAME, something like `${USER\.\-}-v3-8`
1. Spin up a preemptible TPU: `ctpu up -name "$TPU_NAME" -preemptible -tpu-only -tpu-size v3-8`
1. Train the model: `python -m rxrx.main --model-dir "gs://path-to-bucket/trial-id/"` 
1. Watch `tensorboard --logdir=gs://path-to-bucket/`
1. Don't forget to turn off your TPU when you are done `ctpu delete -name "$TPU_NAME"  -tpu-only`
