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

"""Projection operators."""

from typing import Any
from typing import Tuple

import jax
import jax.numpy as jnp

from jaxopt._src.bisection import Bisection
from jaxopt._src.quadratic_prog import QuadraticProgramming
from jaxopt._src import tree_util


def projection_non_negative(x: Any, hyperparams=None) -> Any:
  r"""Projection onto the non-negative orthant.

  The output is ``argmin_{p >= 0} ||x - p||``

  Args:
    x: pytree to project.
    hyperparams: ignored.
  Returns:
    p: projected pytree, same structure as ``x``.
  """
  del hyperparams  # Not used.
  return tree_util.tree_map(jax.nn.relu, x)


def projection_box(x: Any, hyperparams: Tuple) -> Any:
  r"""Projection onto box constraints.

  The output is ``argmin_{lower <= p <= upper} ||x - p||``

  where ``(lower, upper) = hyperparams``.

  Args:
    x: pytree to project.
    hyperparams: a tuple ``(lower, upper)``. ``lower`` and ``upper`` can be
      either scalar values or pytrees of the same structure as ``x``.
  Returns:
    p: projected pytree, same structure as ``x``.
  """
  lower, upper = hyperparams
  return tree_util.tree_multimap(jnp.clip, x, lower, upper)


@jax.custom_jvp
def _projection_unit_simplex(x: jnp.ndarray) -> jnp.ndarray:
  """Projection onto the unit simplex."""
  s = 1.0
  n_features = x.shape[0]
  u = jnp.sort(x)[::-1]
  cssv = jnp.cumsum(u) - s
  ind = jnp.arange(n_features) + 1
  cond = u - cssv / ind > 0
  idx = jnp.count_nonzero(cond)
  threshold = cssv[idx - 1] / idx.astype(x.dtype)
  return jax.nn.relu(x - threshold)


@_projection_unit_simplex.defjvp
def _projection_unit_simplex_jvp(primals, tangents):
  x, = primals
  x_dot, = tangents
  primal_out = _projection_unit_simplex(x)
  supp = primal_out > 0
  card = jnp.count_nonzero(supp)
  tangent_out = supp * x_dot - (jnp.dot(supp, x_dot) / card) * supp
  return primal_out, tangent_out


def projection_simplex(x: jnp.ndarray, s: float = 1.0) -> jnp.ndarray:
  r"""Projection onto the simplex.

  The output is ``argmin_{p : 0 <= p <= s, jnp.sum(p) = s} ||x - p||``

  Args:
    x: vector to project, an array of shape (n,).
    s: value p should sum to (default: 1.0).
  Returns:
    p: projected vector, an array of shape (n,).
  """
  if s is None:
    s = 1.0
  return s * _projection_unit_simplex(x / s)


def projection_l1_sphere(x: jnp.ndarray, radius: float = 1.0) -> jnp.ndarray:
  r"""Projection onto the l1 sphere.

  The output is:
    ``argmin_{y, ||y||_1 = radius} ||y - x||``.

  Args:
    x: array to project.
    radius: radius of the sphere.

  Returns:
    y: output array (same shape as x)
  """
  return jnp.sign(x) * projection_simplex(jnp.abs(x), radius)


def projection_l1_ball(x: jnp.ndarray, radius: float = 1.0) -> jnp.ndarray:
  r"""Projection onto the l1 ball.

  The output is:
    ``argmin_{y, ||y||_1 <= radius} ||y - x||``.

  Args:
    x: array to project.
    radius: radius of the sphere.

  Returns:
    y: output array (same structure as x) normalized to have suitable norm.
  """
  l1_norm = jax.numpy.linalg.norm(x, ord=1)
  return jax.lax.cond(l1_norm <= radius,
                      lambda _: x,
                      lambda _: projection_l1_sphere(x, radius),
                      operand=None)


def projection_l2_sphere(x: Any, radius: float = 1.0) -> Any:
  r"""Projection onto the l2 sphere.

  The output is:
    ``argmin_{y, ||y|| = radius} ||y - x|| = radius * x / ||x||``.

  Args:
    x: pytree to project.
    radius: radius of the sphere.

  Returns:
    y: output pytree (same structure as x) normalized to have suitable norm.
  """
  factor = radius / tree_util.tree_l2_norm(x)
  return tree_util.tree_scalar_mul(factor, x)


def projection_l2_ball(x: Any, radius: float = 1.0) -> Any:
  r"""Projection onto the l2 ball.

  The output is:
    ``argmin_{y, ||y|| <= radius} ||y - x||``.

  Args:
    x: pytree to project.
    radius: radius of the sphere.

  Returns:
    y: output pytree (same structure as x)
  """
  l2_norm = tree_util.tree_l2_norm(x)
  factor = radius / l2_norm
  return jax.lax.cond(l2_norm <= radius,
                      lambda _: x,
                      lambda _: tree_util.tree_scalar_mul(factor, x),
                      operand=None)


def projection_linf_ball(x: Any, radius: float = 1.0) -> Any:
  r"""Projection onto the l-infinity ball.

  The output is:
    ``argmin_{y, ||y||_inf <= radius} ||y - x||``.

  Args:
    x: pytree to project.
    radius: radius of the sphere.

  Returns:
    y: output pytree (same structure as x)
  """
  return projection_box(x, (-radius, radius))


def projection_hyperplane(x: jnp.ndarray, hyperparams: Tuple) -> jnp.ndarray:
  r"""Projection onto a hyperplane.

  The output is:
    ``argmin_{y, dot(a, y) = b} ||y - x||``.

  Args:
    x: pytree to project.
    hyperparams: tuple ``hyperparams = (a, b)``, where ``a`` is a vector and
      ``b`` is a scalar.

  Returns:
    y: output array (same shape as ``x``)
  """
  a, b = hyperparams
  return x - (jnp.dot(a, x) - b) / jnp.dot(a, a) * a


def projection_halfspace(x: jnp.ndarray, hyperparams: Tuple) -> jnp.ndarray:
  r"""Projection onto a halfspace.

  The output is:
    ``argmin_{y, dot(a, y) <= b} ||y - x||``.

  Args:
    x: pytree to project.
    hyperparams: tuple ``hyperparams = (a, b)``, where ``a`` is a vector and
      ``b`` is a scalar.

  Returns:
    y: output array (same shape as ``x``)
  """
  a, b = hyperparams
  return x - jax.nn.relu(jnp.dot(a, x) - b) / jnp.dot(a, a) * a


def projection_affine_set(x: jnp.ndarray, hyperparams: Tuple) -> jnp.ndarray:
  r"""Projection onto an affine set.

  The output is:
    ``argmin_{y, dot(A, y) = b} ||y - x||``.

  Args:
    x: pytree to project.
    hyperparams: tuple ``hyperparams = (A, b)``, where ``A`` is a matrix and
      ``b`` is a vector.

  Returns:
    y: output array (same shape as ``x``)
  """
  # todo: support matvec for A
  A, b = hyperparams
  qp = QuadraticProgramming()
  I = jnp.eye(len(x))
  hyperparams = dict(params_obj=(I, -x), params_eq=(A, b), params_ineq=None)
  return qp.run(**hyperparams).params[0]


def projection_polyhedron(x: jnp.ndarray, hyperparams: Tuple) -> jnp.ndarray:
  r"""Projection onto a polyhedron.

  The output is:
    ``argmin_{y, dot(A, y) = b, dot(G, y) <= h} ||y - x||``.

  Args:
    x: pytree to project.
    hyperparams: tuple ``hyperparams = (A, b, G, h)``, where ``A`` is a matrix,
      ``b`` is a vector, ``G`` is a matrix and ``h`` is a vector.

  Returns:
    y: output array (same shape as ``x``)
  """
  # todo: support matvecs for A and G
  A, b, G, h = hyperparams
  qp = QuadraticProgramming()
  I = jnp.eye(len(x))
  hyperparams = dict(params_obj=(I, -x), params_eq=(A, b), params_ineq=(G, h))
  return qp.run(**hyperparams).params[0]


def _optimality_fun_proj_box_sec(tau, x, hyperparams):
  # An optimal solution has the form
  # p_i = clip(w_i * tau + x_i, alpha_i, beta_i) for all i
  # where tau is the root of fun(tau, hyperparams) = dot(w, p) - c = 0.
  alpha, beta, w, c = hyperparams
  p = jnp.clip(w * tau + x, alpha, beta)
  return jnp.dot(w, p) - c


def _root_proj_box_sec(x, hyperparams):
  alpha, beta, w, _ = hyperparams
  lower = jax.lax.stop_gradient(jnp.min((alpha - x) / w))
  upper = jax.lax.stop_gradient(jnp.max((beta - x) / w))
  bisect = Bisection(optimality_fun=_optimality_fun_proj_box_sec,
                     lower=lower,
                     upper=upper,
                     increasing=True,
                     check_bracket=False)
  return bisect.run(None, x, hyperparams).params


def projection_box_section(x: jnp.ndarray,
                           hyperparams: Tuple,
                           check_feasible: bool = False):
  """Projection onto a box section.

  The projection is::

    argmin_{p : alpha_i <= p_i <= beta_i, jnp.dot(w, p) = c} ||x - p||

  where ``(alpha, beta, w, c) = hyperparams``.

  Args:
    x: vector to project, an array of shape (n,).
    hyperparams: tuple of parameters, ``hyperparams = (alpha, beta, w, c)``,
      where ``w`` is a positive vector. The problem is infeasible if
      dot(w, alpha) > c or if dot(w, beta) < c.
    check_feasible: whether to check feasibility or not.
      If True, function cannot be jitted.
  Returns:
    p: projected vector, an array of shape (n,).
  """

  alpha, beta, w, c = hyperparams

  if check_feasible:
    if jnp.dot(w, alpha) > c:
      raise ValueError("alpha should satisfy dot(w, alpha) <= c")

    if jnp.dot(w, beta) < c:
      raise ValueError("beta should satisfy dot(w, beta) >= c")

  return jnp.clip(w * _root_proj_box_sec(x, hyperparams) + x, alpha, beta)
