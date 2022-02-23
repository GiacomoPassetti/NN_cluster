import jax
import numpy as np


def J_matrix(seed, L):
             print(seed)
             key = jax.random.PRNGKey(seed)
             size_J = int(((L-2)*(L-1)/2) + L - 1)
             J = jax.random.normal(key, shape = (size_J, size_J), dtype = complex)
             norm = np.sqrt(2)*((2*L)**(3/2))
             J = (J + J.T.conj())*4/norm
             return J



