import itertools 
import jax.numpy as jnp
import numpy as np
import jax
from jax import jit
from jax import lax
from jax.numpy.linalg import eigh
from functools import partial
from func_AI import Exact_ground_gen_syk
from func_rh import Exact_ground_gen_random_hop
from func_AI_ph import Exact_ground_gen_syk_particle_hole
import sys

# Defining constants
L = int(sys.argv[1])
seed = int(sys.argv[2])
Exact_ground_gen_syk(L, seed)
"""
for seed in range(10, 20, 1):
    print(seed)
    np.save("Exact energy_L_"+str(L)+"_seed_"+str(seed)+".npy",Exact_ground_gen_syk(L, seed))

"""




