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

from jaxopt._src import base
from jaxopt._src import implicit_diff
from jaxopt._src import linear_solve
from jaxopt._src import loop
from jaxopt._src import loss
from jaxopt._src import objectives
from jaxopt._src import perturbations
from jaxopt._src import projection
from jaxopt._src import prox
from jaxopt._src import tree_util

from jaxopt._src.bisection import Bisection
from jaxopt._src.block_cd import BlockCoordinateDescent
from jaxopt._src.gradient_descent import GradientDescent
from jaxopt._src.mirror_descent import MirrorDescent
from jaxopt._src.optax_wrapper import OptaxSolver
from jaxopt._src.proximal_gradient import ProximalGradient
from jaxopt._src.quadratic_prog import QuadraticProgramming
