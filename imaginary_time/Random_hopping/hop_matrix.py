import jax
import numpy as np


def hop_matrix(seed, L):
             print(seed)
             key = jax.random.PRNGKey(seed)
             size_J = L
             J = jax.random.normal(key, shape = (size_J, size_J), dtype = complex)
             norm = np.sqrt(2)*np.sqrt(L)
             J = (J + J.T.conj())/norm
             return J

for L in [20, 40]:
    for seed in [1]:
        np.save("hop_matrix_L_"+str(L)+"seed_"+str(seed), np.array(hop_matrix(seed, L)))



