# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train a ResNet-50 model on RxRx1 on TPU.

Original file:
    https://github.com/tensorflow/tpu/blob/master/models/official/resnet/resnet_main.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import time
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.contrib import summary
from tensorflow.python.estimator import estimator

from rxrx import input as rxinput
from rxrx.official_resnet import resnet_v1

DEFAULT_INPUT_FN_PARAMS = {
    'tfrecord_dataset_buffer_size': 256,
    'tfrecord_dataset_num_parallel_reads': None,
    'parallel_interleave_cycle_length': 32,
    'parallel_interleave_block_length': 1,
    'parallel_interleave_buffer_output_elements': None,
    'parallel_interleave_prefetch_input_elements': None,
    'map_and_batch_num_parallel_calls': 128,
    'transpose_num_parallel_calls': 128,
    'prefetch_buffer_size': tf.contrib.data.AUTOTUNE,
}

# The mean and stds for each of the channels
GLOBAL_PIXEL_STATS = (np.array([6.74696984, 14.74640167, 10.51260864,
                                10.45369445,  5.49959796, 9.81545561]),
                       np.array([7.95876312, 12.17305868, 5.86172946,
                                 7.83451711, 4.701167, 5.43130431]))


def resnet_model_fn(features, labels, mode, params, n_classes, num_train_images,
                    data_format, transpose_input, train_batch_size,
                    momentum, weight_decay, base_learning_rate,  warmup_epochs,
                    use_tpu, iterations_per_loop, model_dir, tf_precision,
                    resnet_depth):
    """The model_fn for ResNet to be used with TPUEstimator.

    Args:
    features: `Tensor` of batched images
    labels: `Tensor` of labels for the data samples
    mode: one of `tf.estimator.ModeKeys.{TRAIN,EVAL,PREDICT}`
    params: `dict` of parameters passed to the model from the TPUEstimator,
        `params['batch_size']` is always provided and should be used as the
        effective batch size.


    Returns:
        A `TPUEstimatorSpec` for the model
    """
    if isinstance(features, dict):
        features = features['feature']

    # In most cases, the default data format NCHW instead of NHWC should be
    # used for a significant performance boost on GPU/TPU. NHWC should be used
    # only if the network needs to be run on CPU since the pooling operations
    # are only supported on NHWC.
    if data_format == 'channels_first':
        assert not transpose_input  # channels_first only for GPU
        features = tf.transpose(features, [0, 3, 1, 2])

    if transpose_input and mode != tf.estimator.ModeKeys.PREDICT:
        features = tf.transpose(features, [3, 0, 1, 2])  # HWCN to NHWC

    # This nested function allows us to avoid duplicating the logic which
    # builds the network, for different values of --precision.
    def build_network():
        network = resnet_v1(
            resnet_depth=resnet_depth,
            num_classes=n_classes,
            data_format=data_format)
        return network(
            inputs=features, is_training=(mode == tf.estimator.ModeKeys.TRAIN))

    if tf_precision == 'bfloat16':
        with tf.contrib.tpu.bfloat16_scope():
            logits = build_network()
        logits = tf.cast(logits, tf.float32)
    elif tf_precision == 'float32':
        logits = build_network()

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'classes': tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'classify': tf.estimator.export.PredictOutput(predictions)
            })

    # If necessary, in the model_fn, use params['batch_size'] instead the batch
    # size flags (--train_batch_size or --eval_batch_size).
    batch_size = params['batch_size']  # pylint: disable=unused-variable

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    one_hot_labels = tf.one_hot(labels, n_classes)
    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits,
        onehot_labels=one_hot_labels)

    # Add weight decay to the loss for non-batch-normalization variables.
    loss = cross_entropy + weight_decay * tf.add_n([
        tf.nn.l2_loss(v) for v in tf.trainable_variables()
        if 'batch_normalization' not in v.name
    ])

    host_call = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        # Compute the current epoch and associated learning rate from global_step.
        global_step = tf.train.get_global_step()
        steps_per_epoch = tf.cast(num_train_images / train_batch_size, tf.float32)
        current_epoch = (tf.cast(global_step, tf.float32) / steps_per_epoch)
        warmup_steps = warmup_epochs * steps_per_epoch


        period = 10 * steps_per_epoch
        learning_rate = tf.train.cosine_decay_restarts(base_learning_rate,
                                                       global_step,
                                                       period,
                                                       t_mul=1.0,
                                                       m_mul=1.0,
                                                       alpha=0.0,
                                                       name=None)



        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                               momentum=momentum,
                                               use_nesterov=True)

        if use_tpu:
            # When using TPU, wrap the optimizer with CrossShardOptimizer which
            # handles synchronization details between different TPU cores. To the
            # user, this should look like regular synchronous training.
            optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

        # Batch normalization requires UPDATE_OPS to be added as a dependency to
        # the train operation.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)


        def host_call_fn(gs, loss, lr, ce):
            """Training host call. Creates scalar summaries for training metrics.
            This function is executed on the CPU and should not directly reference
            any Tensors in the rest of the `model_fn`. To pass Tensors from the
            model to the `metric_fn`, provide as part of the `host_call`. See
            https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
            for more information.
            Arguments should match the list of `Tensor` objects passed as the second
            element in the tuple passed to `host_call`.
            Args:
            gs: `Tensor with shape `[batch]` for the global_step
            loss: `Tensor` with shape `[batch]` for the training loss.
            lr: `Tensor` with shape `[batch]` for the learning_rate.
            ce: `Tensor` with shape `[batch]` for the current_epoch.
            Returns:
            List of summary ops to run on the CPU host.
            """
            gs = gs[0]
                # Host call fns are executed FLAGS.iterations_per_loop times after one
                # TPU loop is finished, setting max_queue value to the same as number of
                # iterations will make the summary writer only flush the data to storage
                # once per loop.
            with summary.create_file_writer(model_dir,
                                            max_queue=iterations_per_loop).as_default():
                with summary.always_record_summaries():
                    summary.scalar('loss', loss[0], step=gs)
                    summary.scalar('learning_rate', lr[0], step=gs)
                    summary.scalar('current_epoch', ce[0], step=gs)
                    return summary.all_summary_ops()

            # To log the loss, current learning rate, and epoch for Tensorboard, the
            # summary op needs to be run on the host CPU via host_call. host_call
            # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
            # dimension. These Tensors are implicitly concatenated to
            # [params['batch_size']].
        gs_t = tf.reshape(global_step, [1])
        loss_t = tf.reshape(loss, [1])
        lr_t = tf.reshape(learning_rate, [1])
        ce_t = tf.reshape(current_epoch, [1])

        host_call = (host_call_fn, [gs_t, loss_t, lr_t, ce_t])

    else:
        train_op = None

    eval_metrics = None
    if mode == tf.estimator.ModeKeys.EVAL:

        def metric_fn(labels, logits):
            """Evaluation metric function. Evaluates accuracy.
      This function is executed on the CPU and should not directly reference
      any Tensors in the rest of the `model_fn`. To pass Tensors from the model
      to the `metric_fn`, provide as part of the `eval_metrics`. See
      https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
      for more information.
      Arguments should match the list of `Tensor` objects passed as the second
      element in the tuple passed to `eval_metrics`.
      Args:
        labels: `Tensor` with shape `[batch]`.
        logits: `Tensor` with shape `[batch, num_classes]`.
      Returns:
        A dict of the metrics to return from evaluation.
      """
            predictions = tf.argmax(logits, axis=1)
            top_1_accuracy = tf.metrics.accuracy(labels, predictions)
            in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32)
            top_5_accuracy = tf.metrics.mean(in_top_5)

            return {
                'top_1_accuracy': top_1_accuracy,
                'top_5_accuracy': top_5_accuracy,
            }

        eval_metrics = (metric_fn, [labels, logits])

    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        host_call=host_call,
        eval_metrics=eval_metrics)

def main(use_tpu,
         tpu,
         gcp_project,
         tpu_zone,
         url_base_path,
         use_cache,
         model_dir,
         train_epochs,
         train_batch_size,
         num_train_images,
         epochs_per_loop,
         log_step_count_epochs,
         num_cores,
         data_format,
         transpose_input,
         tf_precision,
         n_classes,
         momentum,
         weight_decay,
         base_learning_rate,
         warmup_epochs,
         input_fn_params=DEFAULT_INPUT_FN_PARAMS,
         resnet_depth=50):

    if use_tpu & (tpu is None):
        tpu = os.getenv('TPU_NAME')
    tf.logging.info('tpu: {}'.format(tpu))
    if gcp_project is None:
        gcp_project = os.getenv('TPU_PROJECT')
    tf.logging.info('gcp_project: {}'.format(gcp_project))

    steps_per_epoch = (num_train_images // train_batch_size)
    train_steps = steps_per_epoch * train_epochs
    current_step = estimator._load_global_step_from_checkpoint_dir(model_dir) # pylint: disable=protected-access,line-too-long
    iterations_per_loop = steps_per_epoch * epochs_per_loop
    log_step_count_steps = steps_per_epoch * log_step_count_epochs


    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        tpu if (tpu or use_tpu) else '', zone=tpu_zone, project=gcp_project)


    config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=model_dir,
        save_summary_steps=iterations_per_loop,
        save_checkpoints_steps=iterations_per_loop,
        log_step_count_steps=log_step_count_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=iterations_per_loop,
            num_shards=num_cores,
            per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.
            PER_HOST_V2))  # pylint: disable=line-too-long

    model_fn = functools.partial(
        resnet_model_fn,
        n_classes=n_classes,
        num_train_images=num_train_images,
        data_format=data_format,
        transpose_input=transpose_input,
        train_batch_size=train_batch_size,
        iterations_per_loop=iterations_per_loop,
        tf_precision=tf_precision,
        momentum=momentum,
        weight_decay=weight_decay,
        base_learning_rate=base_learning_rate,
        warmup_epochs=warmup_epochs,
        model_dir=model_dir,
        use_tpu=use_tpu,
        resnet_depth=resnet_depth)


    resnet_classifier = tf.contrib.tpu.TPUEstimator(
        use_tpu=use_tpu,
        model_fn=model_fn,
        config=config,
        train_batch_size=train_batch_size,
        export_to_tpu=False)


    use_bfloat16 = (tf_precision == 'bfloat16')

    train_glob = os.path.join(url_base_path, 'train', '*.tfrecord')

    tf.logging.info("Train glob: {}".format(train_glob))

    train_input_fn = functools.partial(rxinput.input_fn,
            input_fn_params=input_fn_params,
            tf_records_glob=train_glob,
            pixel_stats=GLOBAL_PIXEL_STATS,
            transpose_input=transpose_input,
            use_bfloat16=use_bfloat16)



    tf.logging.info('Training for %d steps (%.2f epochs in total). Current'
                    ' step %d.', train_steps, train_steps / steps_per_epoch,
                    current_step)

    start_timestamp = time.time()  # This time will include compilation time

    resnet_classifier.train(input_fn=train_input_fn, max_steps=train_steps)

    tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                    train_steps, int(time.time() - start_timestamp))


    elapsed_time = int(time.time() - start_timestamp)
    tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                    train_steps, elapsed_time)

    tf.logging.info('Exporting SavedModel.')

    def serving_input_receiver_fn():
        features = {
          'feature': tf.placeholder(dtype=tf.float32, shape=[None, 512, 512, 6]),
        }
        receiver_tensors = features
        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

    resnet_classifier.export_saved_model(os.path.join(model_dir, 'saved_model'), serving_input_receiver_fn)


if __name__ == '__main__':

    p = argparse.ArgumentParser(description='Train ResNet on rxrx1')
    # TPU Parameters
    p.add_argument(
        '--use-tpu',
        type=bool,
        default=True,
        help=('Use TPU to execute the model for training and evaluation. If'
              ' --use_tpu=false, will use whatever devices are available to'
              ' TensorFlow by default (e.g. CPU and GPU)'))
    p.add_argument(
        '--tpu',
        type=str,
        default=None,
        help=(
            'The Cloud TPU to use for training.'
            ' This should be either the name used when creating the Cloud TPU, '
            'or a grpc://ip.address.of.tpu:8470 url.'))
    p.add_argument(
        '--gcp-project',
        type=str,
        default=None,
        help=('Project name for the Cloud TPU-enabled project. '
              'If not specified, we will attempt to automatically '
              'detect the GCE project from metadata.'))
    p.add_argument(
        '--tpu-zone',
        type=str,
        default=None,
        help=('GCE zone where the Cloud TPU is located in. '
              'If not specified, we will attempt to automatically '
              'detect the GCE project from metadata.'))
    p.add_argument('--use-cache', type=bool, default=None)
    # Dataset Parameters
    p.add_argument(
        '--url-base-path',
        type=str,
        default='gs://rxrx1-us-central1/tfrecords/random-42',
        help=('Base path for tfrecord storage bucket url.'))
    # Training parameters
    p.add_argument(
        '--model-dir',
        type=str,
        default=None,
        help=(
            'The Google Cloud Storage bucket where the model and training summaries are'
            ' stored.'))
    p.add_argument(
        '--train-epochs',
        type=int,
        default=1,
        help=(
            'Defining an epoch as one pass through every training example, '
            'the number of total passes through all examples during training. '
            'Implicitly sets the total train steps.'))
    p.add_argument(
        '--num-train-images',
        type=int,
        default=73000
    )
    p.add_argument(
        '--train-batch-size',
        type=int,
        default=512,
        help=('Batch size to use during training.'))
    p.add_argument(
        '--n-classes',
        type=int,
        default=1108,
        help=('The number of label classes - typically will be 1108 '
              'since there are 1108 experimental siRNA classes.'))
    p.add_argument(
        '--epochs-per-loop',
        type=int,
        default=1,
        help=('The number of steps to run on TPU before outfeeding metrics '
              'to the CPU. Larger values will speed up training.'))
    p.add_argument(
        '--log-step-count-epochs',
        type=int,
        default=64,
        help=('The number of epochs at '
              'which global step information is logged .'))
    p.add_argument(
        '--num-cores',
        type=int,
        default=8,
        help=('Number of TPU cores. For a single TPU device, this is 8 because '
              'each TPU has 4 chips each with 2 cores.'))
    p.add_argument(
        '--data-format',
        type=str,
        default='channels_last',
        choices=[
            'channels_first',
            'channels_last',
        ],
        help=('A flag to override the data format used in the model. '
              'To run on CPU or TPU, channels_last should be used. '
              'For GPU, channels_first will improve performance.'))
    p.add_argument(
        '--transpose-input',
        type=bool,
        default=True,
        help=('Use TPU double transpose optimization.'))
    p.add_argument(
        '--tf-precision',
        type=str,
        default='bfloat16',
        choices=['bfloat16', 'float32'],
        help=('Tensorflow precision type used when defining the network.'))

    # Optimizer Parameters

    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument(
        '--base-learning-rate',
        type=float,
        default=0.2,
        help=('Base learning rate when train batch size is 512. '
              'Chosen to match the resnet paper.'))
    p.add_argument(
        '--warmup-epochs',
        type=int,
        default=5,
    )
    args = p.parse_args()
    args = vars(args)
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('Parsed args: ')
    for k, v in args.items():
        tf.logging.info('{} : {}'.format(k, v))
    main(**args)
