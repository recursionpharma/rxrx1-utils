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

Reading individual image files can become an IO bottleneck during training. This is will be a common problem faced by people who use this dataset so we are also releasing an example script to pack the images into TFRecords. We are also making available some pre-created TFRecords available in Google Cloud Storage. The pre-created TFRecords can be found at:

```
gs://rxrx1-us-central1/tfrecords
gs://rxrx1-europe-west4/tfrecords
```

The data lives in these two regional buckets because when you train with TPUs you want to train from buckets in the same region as your TPU. Remember to use the appropriate bucket that is in the same region as your TPU!

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

### Example TPU workflow

First spin up a VM:
```
ctpu up -vm-only -forward-agent -forward-ports -name my-tpu-vm
```

This command will create the VM and `ssh` you into it. Note how the `-vm-only` flag is used. This allows you to spin up the VM separate from the TPU which helps prevent spending money on idle TPUs.

Next, setup the repo and install the dependencies:
```
git clone git@github.com:recursionpharma/rxrx1-utils.git
cd rxrx1-utils
pip install -r requirements.txt # optional if just training!
```

Note that for just training you can skip the `pip install` since the VM will have all the needed deps already.

Next you need to spin up a TPU for training:
```
export TPU_NAME=my-tpu-v3-8
ctpu up -name "$TPU_NAME" -preemptible -tpu-only -tpu-size v3-8
```

Once that is complete you can start a training job:
```
python -m rxrx.main --model-dir "gs://path-to-bucket/trial-id/"
```
You'll also want to launch a `tensorboard` to watch to check the results:

```
tensorboard --logdir=gs://path-to-bucket/
```
Since we used the `-forward-ports` in the `ctpu` command when starting the VM you will be able to view `tensorboard` on your localhost.

Once you are done with the TPU be sure to delete it!
```
ctpu delete -name "$TPU_NAME" -tpu-only`
```

You can then iterate on the code and spin up a TPU again when ready to try again. 

When you are done with your VM you can either stop it or delete it with the `ctpu` command, for example:
```
ctpu delete -name my-tpu-vm
```
