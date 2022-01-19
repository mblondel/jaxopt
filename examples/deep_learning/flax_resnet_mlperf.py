# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Resnet example with Flax and JAXopt.
====================================
"""

import functools

from absl import app
from absl import flags
from datetime import datetime

from functools import partial
from typing import Any, Callable, Sequence, Tuple, Optional

from flax import linen as nn

import jax
import jax.numpy as jnp

from jaxopt import loss
from jaxopt import OptaxSolver
from jaxopt import tree_util

import optax

import tensorflow_datasets as tfds
import tensorflow as tf


import ml_collections


dataset_names = [
    "mnist", "kmnist", "emnist", "fashion_mnist", "cifar10", "cifar100"
]


flags.DEFINE_float("l2reg", 1e-4, "L2 regularization.")
flags.DEFINE_float("learning_rate", 0.2, "Learning rate.")
flags.DEFINE_integer("epochs", 10, "Number of passes over the dataset.")
flags.DEFINE_float("momentum", 0.9, "Momentum strength.")
flags.DEFINE_enum("dataset", "mnist", dataset_names, "Dataset to train on.")
flags.DEFINE_enum("model", "resnet18", ["resnet1", "resnet18", "resnet34"],
                  "Model architecture.")
flags.DEFINE_integer("train_batch_size", 128, "Batch size at train time.")
flags.DEFINE_integer("test_batch_size", 128, "Batch size at test time.")
FLAGS = flags.FLAGS


def load_dataset(split, *, is_training, batch_size):
  version = 3
  ds, ds_info = tfds.load(
      f"{FLAGS.dataset}:{version}.*.*",
      as_supervised=True,  # remove useless keys
      split=split,
      with_info=True)
  ds = ds.cache().repeat()
  if is_training:
    ds = ds.shuffle(10 * batch_size, seed=0)
  ds = ds.batch(batch_size)
  return iter(tfds.as_numpy(ds)), ds_info


from init2winit.model_lib import normalization


def _constant_init(factor):
  def init_fn(key, shape, dtype=jnp.float32):
    del key
    return jnp.ones(shape, dtype) * factor
  return init_fn


class ResidualBlock(nn.Module):
  """Bottleneck ResNet block."""
  filters: int
  strides: Tuple[int, int] = (1, 1)
  axis_name: Optional[str] = None
  axis_index_groups: Optional[Any] = None
  dtype: Any = jnp.float32
  batch_norm_momentum: float = 0.9
  batch_norm_epsilon: float = 1e-5
  bn_output_scale: float = 0.0
  virtual_batch_size: Optional[int] = None
  data_format: Optional[str] = None

  @nn.compact
  def __call__(self, x, train):
    needs_projection = x.shape[-1] != self.filters * 4 or self.strides != (1, 1)
    batch_norm = functools.partial(
        normalization.VirtualBatchNorm,
        momentum=self.batch_norm_momentum,
        epsilon=self.batch_norm_epsilon,
        axis_name=self.axis_name,
        axis_index_groups=self.axis_index_groups,
        dtype=self.dtype,
        virtual_batch_size=self.virtual_batch_size,
        data_format=self.data_format)
    conv = functools.partial(nn.Conv, use_bias=False, dtype=self.dtype)
    residual = x
    if needs_projection:
      residual = conv(
          self.filters * 4, (1, 1), self.strides, name='proj_conv')(residual)
      residual = batch_norm(name='proj_bn')(
          residual, use_running_average=not train)
    y = conv(self.filters, (1, 1), name='conv1')(x)
    y = batch_norm(name='bn1')(y, use_running_average=not train)
    y = nn.relu(y)
    y = conv(self.filters, (3, 3), self.strides, name='conv2')(y)
    y = batch_norm(name='bn2')(y, use_running_average=not train)
    y = nn.relu(y)
    y = conv(self.filters * 4, (1, 1), name='conv3')(y)
    y = batch_norm(
        name='bn3', scale_init=_constant_init(self.bn_output_scale))(
            y, use_running_average=not train)
    y = nn.relu(residual + y)
    return y


class ResNet(nn.Module):
  """ResNetV1."""
  num_classes: int
  num_filters: int = 64
  num_layers: int = 50
  axis_name: Optional[str] = None
  axis_index_groups: Optional[Any] = None
  dtype: Any = jnp.float32
  batch_norm_momentum: float = 0.9
  batch_norm_epsilon: float = 1e-5
  bn_output_scale: float = 0.0
  virtual_batch_size: Optional[int] = None
  data_format: Optional[str] = None

  @nn.compact
  def __call__(self, x, train):
    if self.num_layers not in _block_size_options:
      raise ValueError('Please provide a valid number of layers')
    block_sizes = _block_size_options[self.num_layers]
    conv = functools.partial(nn.Conv, padding=[(3, 3), (3, 3)])
    x = conv(self.num_filters, kernel_size=(7, 7), strides=(2, 2),
             use_bias=False, dtype=self.dtype, name='conv0')(x)
    x = normalization.VirtualBatchNorm(
        momentum=self.batch_norm_momentum,
        epsilon=self.batch_norm_epsilon,
        name='init_bn',
        axis_name=self.axis_name,
        axis_index_groups=self.axis_index_groups,
        dtype=self.dtype,
        virtual_batch_size=self.virtual_batch_size,
        data_format=self.data_format)(x, use_running_average=not train)
    x = nn.relu(x)  # MLPerf-required
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    for i, block_size in enumerate(block_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = ResidualBlock(
            self.num_filters * 2 ** i,
            strides=strides,
            axis_name=self.axis_name,
            axis_index_groups=self.axis_index_groups,
            dtype=self.dtype,
            batch_norm_momentum=self.batch_norm_momentum,
            batch_norm_epsilon=self.batch_norm_epsilon,
            bn_output_scale=self.bn_output_scale,
            virtual_batch_size=self.virtual_batch_size,
            data_format=self.data_format)(x, train=train)
    x = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(self.num_classes, kernel_init=nn.initializers.normal(),
                 dtype=self.dtype)(x)
    return x

# a dictionary mapping the number of layers in a resnet to the number of blocks
# in each stage of the model.
_block_size_options = {
    1: [1],
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
    200: [3, 24, 36, 3]
}


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.learning_rate = 0.1
  config.warmup_epochs = 5.0
  config.momentum = 0.9
  config.batch_size = 128

  config.num_epochs = 100.0
  config.log_every_steps = 100

  # If num_train_steps==-1 then the number of training steps is calculated from
  # num_epochs using the entire dataset. Similarly for steps_per_eval.
  config.num_train_steps = -1
  config.steps_per_eval = -1
  return config


def create_learning_rate_fn(
    config: ml_collections.ConfigDict,
    base_learning_rate: float,
    steps_per_epoch: int):
  """Create learning rate schedule."""
  warmup_fn = optax.linear_schedule(
      init_value=0., end_value=base_learning_rate,
      transition_steps=config.warmup_epochs * steps_per_epoch)
  cosine_epochs = max(config.num_epochs - config.warmup_epochs, 1)
  cosine_fn = optax.cosine_decay_schedule(
      init_value=base_learning_rate,
      decay_steps=cosine_epochs * steps_per_epoch)
  schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, cosine_fn],
      boundaries=[config.warmup_epochs * steps_per_epoch])
  return schedule_fn


def create_cosine_schedule(base_lr, max_training_steps):
  def lr_fn(t):
    decay_factor = (1 + jnp.cos(t / max_training_steps * jnp.pi)) * 0.5
    return base_lr * decay_factor
  return lr_fn


def load_train(dataset):
  for batch in dataset.train_iterator_fn():
    yield batch["inputs"], jnp.argmax(batch["targets"], axis=1)


def load_test(dataset):
  for batch in dataset.test_epoch():
    yield batch["inputs"], jnp.argmax(batch["targets"], axis=1)


def main(argv):
  del argv

  # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')

  from init2winit.dataset_lib import small_image_datasets
  from ml_collections.config_dict import config_dict

  augmented_dataset = small_image_datasets.get_cifar10(
        jax.random.PRNGKey(0),
        FLAGS.train_batch_size,
        FLAGS.test_batch_size,
        config_dict.ConfigDict(
            dict(
                flip_probability=0.5,
                alpha=1.0,
                crop_num_pixels=4,
                use_mixup=False,
                #train_size=45000,
                #valid_size=5000,
                train_size=50000,
                valid_size=0,
                test_size=10000,
                include_example_keys=True,
                input_shape=(32, 32, 3),
                output_shape=(10,))))

  train_ds, ds_info = load_dataset("train", is_training=True,
                                   batch_size=FLAGS.train_batch_size)
  #test_ds, _ = load_dataset("test", is_training=False,
                            #batch_size=FLAGS.test_batch_size)

  input_shape = (1,) + ds_info.features["image"].shape
  num_classes = ds_info.features["label"].num_classes

  iter_per_epoch = ds_info.splits['train'].num_examples // FLAGS.train_batch_size
  iter_per_epoch_test = ds_info.splits['test'].num_examples // FLAGS.test_batch_size

  # Set up model.

  def predict(params, inputs, aux, train=False):
    x = inputs.astype(jnp.float32) / 255.
    all_params = {"params": params, "batch_stats": aux}
    if train:
      # Returns logits and net_state (which contains the key "batch_stats").
      return net.apply(all_params, x, train=True, mutable=["batch_stats"])
    else:
      # Returns logits only.
      return net.apply(all_params, x, train=False)

  logistic_loss = jax.vmap(loss.multiclass_logistic_loss)

  num_filters = 16
  batch_norm_momentum = 0.9
  batch_norm_epsilon = 1e-5
  bn_output_scale = 0.0
  num_layers = 18 # 1, 18, 34, 50, 101, 152, 200
  virtual_batch_size = 64
  data_format = "NHWC"

  net = ResNet(
        num_classes=num_classes,
        num_filters=num_filters,
        num_layers=num_layers,
        batch_norm_momentum=batch_norm_momentum,
        batch_norm_epsilon=batch_norm_epsilon,
        bn_output_scale=bn_output_scale,
        data_format=data_format)

  def loss_from_logits(params, l2reg, logits, labels):
    mean_loss = jnp.mean(logistic_loss(labels, logits))
    sqnorm = tree_util.tree_l2_norm(params, squared=True)
    return mean_loss + 0.5 * l2reg * sqnorm

  def accuracy_and_loss(params, l2reg, data, aux):
    inputs, labels = data
    logits = predict(params, inputs, aux)
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    loss = loss_from_logits(params, l2reg, logits, labels)
    return accuracy, loss

  def loss_fun(params, l2reg, data, aux):
    inputs, labels = data
    logits, net_state = predict(params, inputs, aux, train=True)
    loss = loss_from_logits(params, l2reg, logits, labels)
    # batch_stats will be stored in state.aux
    return loss, net_state["batch_stats"]

  # Initialize solver.
  #config = get_config()
  #base_learning_rate = config.learning_rate * FLAGS.train_batch_size / 256.
  #learning_rate_fn = create_learning_rate_fn(config=config,
                                             #base_learning_rate=base_learning_rate,
                                             #steps_per_epoch=iter_per_epoch)

  train_size = 50000
  num_train_steps = int(300. * train_size / FLAGS.train_batch_size)
  learning_rate_fn = create_cosine_schedule(base_lr=0.1,
                                            max_training_steps=num_train_steps)

  #opt = optax.sgd(learning_rate=FLAGS.learning_rate,
  opt = optax.sgd(learning_rate=learning_rate_fn,
                  momentum=FLAGS.momentum,
                  nesterov=False)

  # We need has_aux=True because loss_fun returns batch_stats.
  solver = OptaxSolver(opt=opt, fun=loss_fun, maxiter=FLAGS.epochs * iter_per_epoch, has_aux=True)

  # Initialize parameters.
  rng = jax.random.PRNGKey(0)
  init_vars = net.init(rng, jnp.zeros(input_shape), train=True)
  params = init_vars["params"]
  batch_stats = init_vars["batch_stats"]
  start = datetime.now().replace(microsecond=0)

  # Run training loop.
  state = solver.init_state(params)
  jitted_update = jax.jit(solver.update)

  train_ds = load_train(augmented_dataset)

  for _ in range(solver.maxiter):
    train_minibatch = next(train_ds)

    if state.iter_num % iter_per_epoch == iter_per_epoch - 1:
      # Once per epoch evaluate the model on the train and test sets.
      train_acc, train_loss = accuracy_and_loss(params, FLAGS.l2reg, train_minibatch, batch_stats)
      test_acc, test_loss = 0., 0.
      # make a pass over test set to compute test accuracy

      test_ds = load_test(augmented_dataset)

      for _ in range(iter_per_epoch_test):
          tmp = accuracy_and_loss(params, FLAGS.l2reg, next(test_ds), batch_stats)
          test_acc += tmp[0] / iter_per_epoch_test
          test_loss += tmp[1] / iter_per_epoch_test

      train_acc = jax.device_get(train_acc)
      train_loss = jax.device_get(train_loss)
      test_acc = jax.device_get(test_acc)
      test_loss = jax.device_get(test_loss)
      # time elapsed without microseconds
      time_elapsed = (datetime.now().replace(microsecond=0) - start)

      print(f"[Epoch {state.iter_num // (iter_per_epoch+1)}/{FLAGS.epochs}] "
            f"Train acc: {train_acc:.3f}, train loss: {train_loss:.3f}. "
            f"Test acc: {test_acc:.3f}, test loss: {test_loss:.3f}. "
            f"Time elapsed: {time_elapsed}")

    params, state = jitted_update(params=params,
                                  state=state,
                                  l2reg=FLAGS.l2reg,
                                  data=train_minibatch,
                                  aux=batch_stats)
    batch_stats = state.aux


if __name__ == "__main__":
  app.run(main)
