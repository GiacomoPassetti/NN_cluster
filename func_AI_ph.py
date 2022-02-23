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
    return single_trans(states, L, N)

def num_of_trans(N):
  return 1+(N**2)+(((N**2)*((N-1)**2))/4)

     
@partial(jax.vmap, in_axes = (0, None, None))
def double_trans_jax(states, L, N):
      """
        This calculates all double transitions for the syk model by using the single_trans function twice
      Parameters
      ----------
      states : jnp.array
          The batched input states. Expects shape (batch, L)
      Returns
      -------
      trans_states : jnp.array
          All states where two particle have been transferred (or a single particle twice), it already returns 
          the unique transitions
          Output shape = (batch, num_trans(N), L)sa
      """
      # first we generate the one-particle transitions


      unique_trans = jnp.zeros(( int(num_of_trans(N)), L))
      sn = single_trans(states, L, N)
      
      double_trans = single_trans_jax(sn, L, N)
      
      double_trans = double_trans.reshape(N**4, L )
      unique_trans = unique_trans.at[:].set(jnp.unique(double_trans[:], axis = 0, size = int(num_of_trans(N)) ))
 
      return  unique_trans

def single_trans(states, L, N):
      
      transitions = jnp.zeros((N*N,L))
      generator = jnp.ones_like(states) - states
      ind_an = jnp.repeat(jnp.array(jnp.nonzero(states, size = N)), N)
      ind_gen = jnp.tile(jnp.array(jnp.nonzero(generator, size = N)), N)
      annihilator = transitions.at[jnp.arange(N*N), ind_an].set(-1)
      generator = transitions.at[jnp.arange(N*N), ind_gen].set(1)
      transitions = jnp.repeat(states.reshape(1, L), repeats = N*N, axis = 0)
      transitions = transitions + generator + annihilator
      

      return transitions


#given two "real space" indeces i and j it returns the associate J_ij,lk index.  
@jit
def I_J_conv(i, j):
    return (((j-1)*(j)/2) + i).astype(int)




partial(jax.jit, static_argnums=(1,2))
def Jordan_Wigner_OP(v, i, k):
                 zeros = jnp.zeros(v.shape[-1])
                 mask = jnp.arange(v.shape[-1])
                 mask =  (mask >= i) & (mask < k)
 
                 return  (-1)**(jnp.sum(jnp.where(mask, v, zeros) ))
                 


def zeroth_energy_experiment(v, J, N):
     
     ind = jnp.where(v == 1, size = N)[0]
     H = 0
     
     for k in range(1, N, 1):
         for i in range(k):


             sgn1 = Jordan_Wigner_OP(v, ind[i], ind[k] )

             v_intermediate = v.at[ind[i]].set(0)
             v_intermediate = v_intermediate.at[ind[k]].set(0)

             sgn2 = Jordan_Wigner_OP(v_intermediate, ind[i], ind[k] )

             H += sgn1*sgn2*J[I_J_conv(ind[i], ind[k]), I_J_conv(ind[i], ind[k])]
     return H


def first_ord_energy(v1, v2, J, N):

       des_ind = jnp.where((v1-v2) == 1, size = 1)[0]
       cre_ind = jnp.where((v1-v2) == -1, size = 1)[0]
       
       static_ind = jnp.where(jnp.multiply(v1, v2)==1, size = N -1)[0]

       H = 0
       for i in range(static_ind.shape[0]):
           
           dx_ind = jnp.sort(jnp.array([des_ind[0], static_ind[i]]))
           sx_ind = jnp.sort(jnp.array([cre_ind[0], static_ind[i]]))
           sgn_ph = ((-1)**( jnp.array([des_ind[0], static_ind[i]])[0]==dx_ind[0]).astype(int))*((-1)**(jnp.array([cre_ind[0], static_ind[i]])[0]==sx_ind[0]).astype(int))
           sgn_JW_ph = Jordan_Wigner_OP(v1, 0, des_ind[0])*Jordan_Wigner_OP(v1.at[des_ind[0]].set(0), 0, cre_ind[0])
           

           sgn1 = Jordan_Wigner_OP(v1, dx_ind[0], dx_ind[1])
       
           v_intermediate = v1.at[des_ind[0]].set(0)
           v_intermediate = v_intermediate.at[static_ind[0]].set(0)
       

           sgn2 = Jordan_Wigner_OP(v_intermediate, sx_ind[0], sx_ind[1])
           H += ((sgn1*sgn2)+(0.5*(sgn_ph*sgn_JW_ph)))*J[I_J_conv(sx_ind[0], sx_ind[1]), I_J_conv(dx_ind[0], dx_ind[1])]
       return H
       

def second_ord_energy(v1, v2, J, L, N):

      dx_ind = jnp.where((v1-v2) == 1, size = 2)[0]
      sx_ind = jnp.where((v1-v2) == -1, size = 2)[0]

      sgn1 = Jordan_Wigner_OP(v1, dx_ind[0], dx_ind[1])
      v_intermediate = v1.at[dx_ind[0]].set(0)
      v_intermediate = v_intermediate.at[dx_ind[1]].set(0)

      sgn2 = Jordan_Wigner_OP(v_intermediate, sx_ind[0], sx_ind[1])
      

      return sgn1*sgn2*J[I_J_conv(sx_ind[0], sx_ind[1]), I_J_conv(dx_ind[0], dx_ind[1])]

@partial(jax.vmap, in_axes = (0, 0, None, None, None))
def energy_elements_syk(v, trans, J, L, N):
      H_syk = jnp.zeros(int(num_of_trans(N)), dtype = complex)

      transitions_map = jnp.sum(jnp.abs(jnp.tile(v, (int(num_of_trans(N)), 1))-trans), axis = 1)

      zeroth = jnp.where(transitions_map == 0, size = 1)[0]
      first = jnp.where(transitions_map == 2, size = N*N)[0]
      second = jnp.where(transitions_map == 4, size = int(num_of_trans(N)) - N*N -1)[0]



      H_syk = H_syk.at[zeroth[0]].set(zeroth_energy_experiment(v, J, N))
      for i in range(first.shape[0]):
          H_syk = H_syk.at[first[i]].set(first_ord_energy(v, trans[first[i]], J, N))
      for i in range(second.shape[0]):
          H_syk = H_syk.at[second[i]].set(second_ord_energy(v, trans[second[i]], J, L, N))


      return H_syk





def Exact_ground_gen_syk_particle_hole(L, seed):
         N = int(L/2)
         key = jax.random.PRNGKey(seed)
         size_J = int(((L-2)*(L-1)/2) + L - 1)
         J = jax.random.normal(key, shape = (size_J, size_J), dtype = complex)

         norm = np.sqrt(4 * size_J)*((2*L)**(3/2))
         J = (J + J.T.conj())*4/norm
         
         states = jnp.array(states_gen(L, N))

         trans = double_trans_jax(states, L, N)

         states_conv, trans_conv = mat_conv(states, trans, L, N, 0)
         
         seed_mat = energy_elements_syk(states, trans, J, L, N)
         
         H_SYK = jnp.zeros((states.shape[0], states.shape[0]), dtype = complex)
         
         for i in range(states.shape[0]):
             for j in range(trans_conv.shape[1]):
                   ind_column = np.where(states_conv == trans_conv[i, j])
                   a = ind_column[0]
         
                   H_SYK = H_SYK.at[i, a].set(seed_mat[i, j])
         
         
                   
         u, v = eigh(H_SYK)
         return u[0]/L


