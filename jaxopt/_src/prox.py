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

"""Proximity operators."""

from typing import Any
from typing import Optional
from typing import Tuple

import jax
import jax.numpy as jnp

from jaxopt._src import tree_util


def prox_none(x: Any, hyperparams: Optional[Any] = None, scaling: float = 1.0):
  r"""Proximal operator for g(x) = 0, i.e., the identity function.

  Since g(x) = 0, the output is: ``argmin_y 0.5 ||y - x||^2 = Id(x)``.

  Args:
    x: input pytree.
    hyperparams: ignored.
    scaling: ignored.
  Returns:
    y: output pytree with same structure as x.
  """
  del hyperparams, scaling
  return x


def prox_lasso(x: Any, hyperparams: Any, scaling: float = 1.0):
  r"""Proximal operator for the l1 norm, i.e., soft-thresholding operator.

  The output is:
    argmin_y 0.5 ||y - x||^2 + scaling * hyperparams * ||y||_1.
  When hyperparams is a pytree, the weights are applied coordinate-wise.

  Args:
    x: input pytree.
    hyperparams: regularization strength, float or pytree (same structure as x).
    scaling: a scaling factor.

  Returns:
    y: output pytree with same structure as x.
  """
  fun = lambda u, v: jnp.sign(u) * jax.nn.relu(jnp.abs(u) - v * scaling)
  return tree_util.tree_multimap(fun, x, hyperparams)


def prox_non_negative_lasso(x, hyperparam=1.0, scaling=1.0):
  r"""Proximal operator for the l1 norm on the non-negative orthant.

  The output is:
    argmin_{y >= 0} 0.5 ||y - x||^2 + scaling * hyperparam * ||y||_1.

  Args:
    x: input pytree.
    hyperparam: regularization strength, float.
    scaling: a scaling factor.

  Returns:
    y: output pytree with same structure as x.
  """
  pytree = tree_util.tree_add(x, -hyperparam * scaling)
  return tree_util.tree_map(jax.nn.relu, pytree)


def prox_elastic_net(x: Any,
                     hyperparams: Tuple[Any, Any],
                     scaling: float = 1.0):
  r"""Proximal operator for the elastic net.

  The output is:
    argmin_y 0.5 ||y - x||^2 + scaling * hyperparams[0] * g(y)

  where g(y) = ||y||_1 + hyperparams[1] * 0.5 * ||y||_2^2.

  Args:
    x: input pytree.
    hyperparams: a tuple, where both hyperparams[0] and hyperparams[1] can be
      either floats or pytrees with the same structure as x.
    scaling: a scaling factor.

  Returns:
    y: output pytree with same structure as x.
  """
  prox_l1 = lambda u, lam: jnp.sign(u) * jax.nn.relu(jnp.abs(u) - lam)
  fun = lambda u, lam, gamma: (prox_l1(u, scaling * lam) /
                               (1.0 + scaling * lam * gamma))
  return tree_util.tree_multimap(fun, x, hyperparams[0], hyperparams[1])


def prox_group_lasso(x: Any, hyperparam: float, scaling=1.0):
  r"""Proximal operator for the l2 norm, i.e., block soft-thresholding operator.

  The output is:
    argmin_y 0.5 ||y - x||^2 + scaling * hyperparam * ||y||_2.

  Blocks can be grouped using ``jax.vmap``.

  Args:
    x: input pytree.
    hyperparam: regularization strength, float.
    scaling: a scaling factor.

  Returns:
    y: output pytree with same structure as x.
  """
  l2_norm = tree_util.tree_l2_norm(x)
  factor = 1 - hyperparam * scaling / l2_norm
  factor = jnp.where(factor >= 0, factor, 0)
  return tree_util.tree_scalar_mul(factor, x)


def prox_ridge(x: Any, hyperparam: float, scaling=1.0):
  r"""Proximal operator for the squared l2 norm.

  The output is:
    argmin_y 0.5 ||y - x||^2 + scaling * hyperparam * 0.5 * ||y||_2^2.

  Args:
    x: input pytree.
    hyperparam: regularization strength, float.
    scaling: a scaling factor.

  Returns:
    y: output pytree with same structure as x.
  """
  factor = 1. / (1 + scaling * hyperparam)
  return tree_util.tree_scalar_mul(factor, x)


def prox_non_negative_ridge(x, hyperparam=1.0, scaling=1.0):
  r"""Proximal operator for the squared l2 norm on the non-negative orthant.

  The output is:
    argmin_{y >= 0} 0.5 ||y - x||^2 + scaling * hyperparam * 0.5 * ||y||_2^2.

  Args:
    x: input pytree.
    hyperparam: regularization strength, float.
    scaling: a scaling factor.

  Returns:
    y: output pytree with same structure as x.
  """
  pytree = tree_util.tree_scalar_mul(1./ (1 + hyperparam * scaling), x)
  return tree_util.tree_map(jax.nn.relu, pytree)


def make_prox_from_projection(projection):
  def prox(x, hyperparams=None, scaling=1.0):
    del scaling  # The scaling parameter is meaningless for projections.
    return projection(x, hyperparams)
  return prox
