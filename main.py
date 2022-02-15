import itertools 
import jax.numpy as jnp
import numpy as np
import jax
from jax import jit
from jax import lax
from jax.numpy.linalg import eigh
from functools import partial
from func_AI import Exact_ground_gen, double_trans_jax, energy_elements, states_gen
import sys

# Defining constants
L = int(sys.argv[1])
N = int(L/2)
v = jnp.zeros(L)
seed = int(sys.argv[2])


print(Exact_ground_gen(L, seed))

