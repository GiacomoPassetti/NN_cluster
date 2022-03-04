import netket as nk
from netket.operator import AbstractOperator
import numpy as np
import jax.numpy as jnp
import jax
from jax import jit
from functools import partial
from netket.hilbert import Fock, Spin
from netket.vqs.mc.kernels import batch_discrete_kernel
from jax.numpy.linalg import eigh
import netket.experimental as nkx
from J_matrix import J_matrix
from netket.models import Jastrow
import itertools
import sys
import time

L = int(sys.argv[1])
N = int(L/2)
alpha = int(sys.argv[2])
seed = int(sys.argv[3])
steps = int(sys.argv[4])
samples = int(sys.argv[5])
lr = [0.1, 0.01, 0.005, 0.001]
J = jnp.array(np.load("J_matrix_L_"+str(L)+"seed_"+str(seed)+".npy"))
#print(J)

class XOperator(AbstractOperator):
  @property
  def dtype(self):
    return float
  @property
  def is_hermitian(self):
    return True


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

@jax.vmap
def single_trans_jax(states):
    return single_trans(states, L, N)

def num_of_trans(N):
  return 1+(N**2)+(((N**2)*((N-1)**2))/4)

     
#@partial(jax.vmap, in_axes = (0, None, None))
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
      
      double_trans = single_trans_jax(sn)
      
      double_trans = double_trans.reshape(N**4, L )
      unique_trans = unique_trans.at[:].set(jnp.unique(double_trans[:], axis = 0, size = int(num_of_trans(N)) ))
      return unique_trans
      
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

           

           sgn1 = Jordan_Wigner_OP(v1, dx_ind[0], dx_ind[1])

           v_intermediate = v1.at[dx_ind[0]].set(0)
           v_intermediate = v_intermediate.at[dx_ind[1]].set(0)
           v_intermediate = v_intermediate.at[sx_ind[1]].set(1)

           
           sgn2 = Jordan_Wigner_OP(v_intermediate, sx_ind[0], sx_ind[1])

           H += sgn1*sgn2*J[I_J_conv(sx_ind[0], sx_ind[1]), I_J_conv(dx_ind[0], dx_ind[1])]
       return H
       

def second_ord_energy(v1, v2, J, L, N):

      dx_ind = jnp.where((v1-v2) == 1, size = 2)[0]
      sx_ind = jnp.where((v1-v2) == -1, size = 2)[0]

      sgn1 = Jordan_Wigner_OP(v1, dx_ind[0], dx_ind[1])
      v_intermediate = v1.at[dx_ind[0]].set(0)
      v_intermediate = v_intermediate.at[dx_ind[1]].set(0)

      sgn2 = Jordan_Wigner_OP(v_intermediate, sx_ind[0], sx_ind[1])
      

      return sgn1*sgn2*J[I_J_conv(sx_ind[0], sx_ind[1]), I_J_conv(dx_ind[0], dx_ind[1])]

    
#@partial(jax.vmap, in_axes = (0, 0, None, None))
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
        

@partial(jax.vmap, in_axes=(None, None, 0, None), out_axes=0)
def e_loc(logpsi, pars, sigma, _extra_args):
    eta, mels = get_conns_and_mels(sigma)
    return jnp.sum(mels * jnp.exp(logpsi(pars, eta) - logpsi(pars, sigma)), axis=-1)


@nk.vqs.get_local_kernel.dispatch
def get_local_kernel(_vstate: nk.vqs.MCState, _op: XOperator):
    return e_loc


@nk.vqs.get_local_kernel_arguments.dispatch
def get_local_kernel_arguments(vstate: nk.vqs.MCState, _op: XOperator):
    return vstate.samples, ()


def get_conns_and_mels(sigma):
    # create the possible transitions from the state sigma
    eta = double_trans_jax(sigma, L, N)

   
    # generate deterministically the associated entries of the syk hamiltonian
    mels = energy_elements_syk(sigma, eta, J, L, N)
    
    return eta, mels





my_graph = nk.graph.Chain(length = L)
hi = Fock(n_max=1, n_particles=N, N=L)
X_OP = XOperator(hi)









#optimizer = nk.optimizer.Adam()
optimizer = nk.optimizer.Sgd(learning_rate = 0.1)

#print("For this value of the seed the exact GS energy is : ", Exact_ground(seed))
#model = Module()
model = nk.models.RBM(dtype = complex, alpha = alpha)
#model = Jastrow()
#model = nk.models.ARNNConv2D()
vs  = nk.vqs.MCState(nk.sampler.MetropolisExchange(hilbert =hi, graph = my_graph, d_max = L),model = model , n_samples = samples)
sr = nk.optimizer.SR()
gs = nk.VMC(hamiltonian = X_OP, optimizer = optimizer, variational_state=vs, preconditioner=sr)



t0 = time.time()
ID = "J_constant_run_L_"+str(L)+"seed_"+"{:.2f}".format(seed)+"alpha_"+"{:.2f}".format(alpha)+"samples_"+str(samples)

logs = [
    nk.logging.JsonLog(ID, save_params=True, save_params_every=1),
    #nk.logging.StateLog("state_checkpoint_constant"+ID, tar=True),
]

"""
for step in range(len(lr)):
   gs.optimizer = nk.optimizer.Sgd(learning_rate = lr[step])
   gs.run(steps, out = logs)
"""
t0 = time.time()
gs.run(1 , out = logs)
print("before compilation", time.time() - t0)
t1 = time.time()
gs.run(steps , out = logs)
print("after compilation", time.time() - t1)






