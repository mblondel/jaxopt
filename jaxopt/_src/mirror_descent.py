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

"""Implementation of mirror descent in JAX."""

from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional
from typing import Union

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jaxopt._src import base
from jaxopt._src import implicit_diff as idf
from jaxopt._src import linear_solve
from jaxopt._src import loop
from jaxopt._src.tree_util import tree_add_scalar_mul
from jaxopt._src.tree_util import tree_l2_norm
from jaxopt._src.tree_util import tree_sub


class MirrorDescentState(NamedTuple):
  """Named tuple containing state information."""
  iter_num: int
  error: float


@dataclass
class MirrorDescent:
  """Mirror descent solver.

  This solver minimizes::
    argmin_x fun(x, *args, **kwargs),
  where fun is smooth with convex domain.

  The stopping criterion is::
    ||x - projection_grad(x, g, 1.0, hyperparams_proj)||_2 <= tol,
  where ``g = grad(fun)(x, *args, **kwargs)``.

  Attributes:
    fun: a smooth function of the form ``fun(x, *args, **kwargs)``.
    projection_grad: a function of the form
      ``projection_grad(x, g, stepsize, hyperparams_proj)`` representing the
      mirror descent update for iterate x and gradient g. Optionally, it can be
      instantiated from a projection and mapping function (mirror map) using the
      method `make_projection_grad`.
    stepsize: a stepsize to use, or a callable specifying the stepsize to use at
      each iteration.
    maxiter: maximum number of mirror descent iterations.
    tol: tolerance to use.
    verbose: whether to print error on every iteration or not. verbose=True will
      automatically disable jit.
    implicit_diff: if True, enable implicit differentiation using cg,
      if Callable, do implicit differentiation using callable as linear solver,
      if False, use autodiff through the solver implementation (note:
        this will unroll syntactic loops).
    has_aux: whether function fun outputs one (False) or more values (True).
      When True it will be assumed by default that fun(...)[0] is the objective.

  References:
    Nemirovskij, Arkadij Semenovič, and David Borisovich Yudin. "Problem
    complexity and method efficiency in optimization." J. Wiley @ Sons, New
    York(1983).
  """
  fun: Callable
  projection_grad: Optional[Callable]
  stepsize: Union[float, Callable]
  maxiter: int = 500
  tol: float = 1e-2
  verbose: int = 0
  implicit_diff: Union[bool, Callable] = False
  has_aux: bool = False

  @staticmethod
  def make_projection_grad(projection: Callable,
                           mapping_fun: Callable) -> Callable:
    """Instantiates `projection_grad` argument from projection and mirror map.

    Args:
      projection: projection operator of the form
        ``projection(x, hyperparams_proj)``, typically
        ``argmin_z D_{gen_fun}(z, mapping_fun^{-1}(y))``.
      mapping_fun: the mirror map, typically of the form
        ``mapping_fun = grad(gen_fun)``, where `gen_fun` is the generating
        function of the Bregman divergence.
    Return type:
      Callable
    Returns:
      A function `projection_grad(x, g, stepsize, hyperparams_proj)`
      representing the mirror descent update for iterate x and gradient g.
    """
    def projection_grad(x, x_fun_grad, stepsize, hyperparams_proj):
      update = tree_add_scalar_mul(mapping_fun(x), -stepsize, x_fun_grad)
      return projection(update, hyperparams_proj)
    return projection_grad

  def init(self,
           init_params: Any) -> base.OptStep:
    """Initialize the ``(params, state)`` pair.

    Args:
      init_params: pytree containing the initial parameters.
    Return type:
      base.OptStep
    Returns:
      (params, state)
    """
    state = MirrorDescentState(iter_num=0, error=jnp.inf)
    return base.OptStep(params=init_params, state=state)

  def _error(self, x, x_fun_grad, hyperparams_proj):
    next_x = self.projection_grad(x, x_fun_grad, 1.0, hyperparams_proj)
    diff_x = tree_sub(next_x, x)
    return tree_l2_norm(diff_x)

  def _update(self, x, state, hyperparams_proj, args, kwargs):
    iter_num, _ = state
    stepsize = (self.stepsize(iter_num) if isinstance(self.stepsize, Callable)
                else self.stepsize)
    x_fun_grad = self._grad_fun(x, *args, **kwargs)
    next_x = self.projection_grad(x, x_fun_grad, stepsize, hyperparams_proj)
    error = self._error(x, x_fun_grad, hyperparams_proj)
    next_state = MirrorDescentState(iter_num=iter_num + 1, error=error)
    return base.OptStep(params=next_x, state=next_state)

  def update(self,
             params: Any,
             state: NamedTuple,
             hyperparams_proj: Any,
             *args,
             **kwargs) -> base.OptStep:
    """Performs one iteration of mirror descent.

    Args:
      params: pytree containing the parameters.
      state: named tuple containing the solver state.
      hyperparams_proj: pytree containing hyperparameters of projection.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.
    Return type:
      base.OptStep
    Returns:
      (params, state)
    """
    f = self._update
    return f(params, state, hyperparams_proj, args, kwargs)

  def run(self,
          init_params: Any,
          hyperparams_proj: Any,
          *args,
          **kwargs) -> base.OptStep:
    """Runs mirror descent until convergence or max number of iterations.

    Args:
      init_params: pytree containing the initial parameters.
      hyperparams_proj: pytree containing hyperparameters of projection.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.
    Return type:
      base.OptStep
    Returns:
      (params, state)
    """
    def cond_fun(pair):
      _, state = pair
      if self.verbose:
        print(state.iter_num, state.error)
      return state.error > self.tol

    def body_fun(pair):
      params, state = pair
      return self.update(params, state, hyperparams_proj, *args, **kwargs)

    return loop.while_loop(cond_fun=cond_fun, body_fun=body_fun,
                           init_val=self.init(init_params),
                           maxiter=self.maxiter, jit=self._jit,
                           unroll=self._unroll)

  def _fixed_point_fun(self, sol, hyperparams_proj, args, kwargs):
    sol_fun_grad = self._grad_fun(sol, *args, **kwargs)
    return self.projection_grad(sol, sol_fun_grad, 1.0, hyperparams_proj)

  def optimality_fun(self, sol, hyperparams_proj, *args, **kwargs):
    """Optimality function mapping compatible with ``@custom_root``."""
    fp = self._fixed_point_fun(sol, hyperparams_proj, args, kwargs)
    return tree_sub(fp, sol)

  def __post_init__(self):
    if self.has_aux:
      self.fun = jax.jit(lambda x, par: self.fun(x, par)[0])
    else:
      self.fun = jax.jit(self.fun)

    # Pre-compile useful functions.
    self._grad_fun = jax.jit(jax.grad(self.fun))

    # We always jit unless verbose mode is enabled.
    self._jit = not self.verbose
    # We unroll when implicit diff is disabled or when jit is disabled.
    self._unroll = not self.implicit_diff or not self._jit

    # Set up implicit diff.
    if self.implicit_diff:
      if isinstance(self.implicit_diff, Callable):
        solve = self.implicit_diff
      else:
        solve = linear_solve.solve_normal_cg

      decorator = idf.custom_root(self.optimality_fun,
                                  has_aux=True,
                                  solve=solve)
      # pylint: disable=g-missing-from-attributes
      self.run = decorator(self.run)
