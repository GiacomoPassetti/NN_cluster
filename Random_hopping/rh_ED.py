import itertools 
import jax.numpy as jnp
import numpy as np
import jax
from jax import jit
from jax import lax
from jax.numpy.linalg import eigh
from functools import partial

def states_gen(L,N):
    which = np.array(list(itertools.combinations(range(L), N)))
    #print(which)
    grid = np.zeros((len(which), L), dtype="int8")

    # Magic
    grid[np.arange(len(which))[None].T, which] = 1
    
    return grid

@partial(jax.vmap, in_axes = (0, 0, None, None, None))
def mat_conv(states, transitions, L, N, seed):
        bin = jnp.flipud(2**jnp.arange(0, L, 1))
        
        NMax = (jnp.tensordot(jnp.ones(L),bin, axes = ([0], [0])))
        
        
        # The transitions tensor get converted to the decimal base by contracting their L dimension. States gets repeated num_trans times to allow the states sorting. 

        trans_converted = jnp.tensordot(transitions, bin, axes = ([1], [0]))

        states_conv = jnp.tensordot(states, bin, 1)
        return states_conv, trans_converted

@partial(jax.vmap, in_axes = (0, None, None))
def single_trans_jax(states, L, N):
    return single_trans_rh(states, L, N)

def single_trans_rh(states, L, N):
      
      transitions = jnp.zeros(((N*N),L))
      generator = jnp.ones_like(states) - states
      ind_an = jnp.repeat(jnp.array(jnp.nonzero(states, size = N)), N)
      ind_gen = jnp.tile(jnp.array(jnp.nonzero(generator, size = N)), N)
      annihilator = transitions.at[jnp.arange(N*N), ind_an].set(-1)
      generator = transitions.at[jnp.arange(N*N), ind_gen].set(1)
      
      transitions = jnp.repeat(states.reshape(1, L), repeats = N*N, axis = 0)
      transitions = transitions + generator + annihilator
      transitions = jnp.concatenate((states.reshape(1, L), transitions), axis = 0)
      

      return transitions


partial(jax.jit, static_argnums=(1,2))
def Jordan_Wigner_OP(v, i, k):
                 zeros = jnp.zeros(v.shape[-1])
                 mask = jnp.arange(v.shape[-1])
                 mask =  (mask >= i) & (mask < k)
 
                 return  (-1)**(jnp.sum(jnp.where(mask, v, zeros) ))


def zeroth_energy_experiment(v, J, N):
     
     ind = jnp.where(v == 1, size = N)[0]
     H = 0
     for i in range(N):
         H += J[ind[i], ind[i]]
     return H


def first_ord_energy_random_hopping(v1, v2, J, N):

       des_ind = jnp.where((v1-v2) == 1, size = 1)[0]
       cre_ind = jnp.where((v1-v2) == -1, size = 1)[0]
       
       sgn1 = Jordan_Wigner_OP(v1, 0, des_ind[0])
       v_intermediate = v1.at[des_ind[0]].set(0)
       sgn2 = Jordan_Wigner_OP(v_intermediate, 0, cre_ind[0])

       return sgn1*sgn2*J[cre_ind[0], des_ind[0]]

@partial(jax.vmap, in_axes = (0, 0, None, None, None))
def energy_elements_random_hop(v, trans, J, L, N):
      H_rh = jnp.zeros(((N*N)+1), dtype = complex)
      H_rh = H_rh.at[0].set(zeroth_energy_experiment(v, J, N))
      for i in range(N*N):
          H_rh = H_rh.at[i+1].set(first_ord_energy_random_hopping(v, trans[i+1], J, N))
      
      return H_rh

def Exact_ground_gen_random_hop(L, J,seed):
         N = int(L/2)
         states = jnp.array(states_gen(L, N))

         trans = single_trans_jax(states, L, N)

         states_conv, trans_conv = mat_conv(states, trans, L, N, 0)
         
         seed_mat = energy_elements_random_hop(states, trans, J, L, N)
         
         H_rh = jnp.zeros((states.shape[0], states.shape[0]), dtype = complex)
         
         for i in range(states.shape[0]):
             for j in range(trans_conv.shape[1]):
                   ind_column = np.where(states_conv == trans_conv[i, j])
                   a = ind_column[0]
         
                   H_rh = H_rh.at[i, a].set(seed_mat[i, j])
         assert np.allclose(H_rh, H_rh.conj().T)
         u, v = eigh(H_rh)
         return u[0]


for L in [4, 6, 8, 10]:
       print(L)
       seed = 1
       J = jnp.array(np.load("hop_matrix_L_"+str(L)+"seed_"+str(seed)+".npy"))
       E = Exact_ground_gen_random_hop(L, J, seed)
       print("ED energy: ", E)
       np.save("ED_energy_L"+str(L)+"seed_"+str(seed)+".npy", E)