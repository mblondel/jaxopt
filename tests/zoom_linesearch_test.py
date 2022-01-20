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

from absl.testing import absltest

import jax
from jax import test_util as jtu
import jax.numpy as jnp

from jaxopt import objective
from jaxopt._src import test_util
from jaxopt import ZoomLineSearch

import numpy as onp

from sklearn import datasets


class ZoomLinesearchTest(jtu.JaxTestCase):

  def test_zoom_linesearch_array(self):
    X, y = datasets.load_digits(return_X_y=True)
    y = jnp.where(y >= 1, 1, 0)
    data = (X, y)
    fun = objective.binary_logreg

    rng = onp.random.RandomState(4)
    w_init = rng.randn(X.shape[1])

    # Call to run.
    ls = ZoomLineSearch(fun=fun, maxiter=20)
    stepsize, state = ls.run(init_stepsize=1.0, params=w_init, data=data)

    print(stepsize)

  def test_zoom_linesearch_pytree(self):
    X, y = datasets.load_digits(return_X_y=True)
    data = (X, y)
    n_classes = len(jnp.unique(y))
    fun = objective.multiclass_logreg_with_intercept

    rng = onp.random.RandomState(4)
    W_init = rng.randn(X.shape[1], n_classes)
    b_init = rng.randn(n_classes)
    pytree_init = (W_init, b_init)

    # Call to run.
    ls = ZoomLineSearch(fun=fun, maxiter=20)
    stepsize, state = ls.run(init_stepsize=1.0, params=pytree_init, data=data)

    print(stepsize)



if __name__ == '__main__':
  # Uncomment the line below in order to run in float64.
  jax.config.update("jax_enable_x64", True)
  absltest.main(testLoader=jtu.JaxTestLoader())
