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

from netket.models import Jastrow
import itertools
import sys
import time



class XOperator(AbstractOperator):
  @property
  def dtype(self):
    return float
  @property
  def is_hermitian(self):
    return True


def single_trans_hop(states, L, N):
      
      transitions = jnp.zeros((N*N,L))
      generator = jnp.ones_like(states) - states
      ind_an = jnp.repeat(jnp.array(jnp.nonzero(states, size = N)), N)
      ind_gen = jnp.tile(jnp.array(jnp.nonzero(generator, size = N)), N)
      annihilator = transitions.at[jnp.arange(N*N), ind_an].set(-1)
      generator = transitions.at[jnp.arange(N*N), ind_gen].set(1)
      transitions = jnp.repeat(states.reshape(1, L), repeats = N*N, axis = 0)
      transitions = transitions + generator + annihilator
      transitions = jnp.concatenate((states.reshape((1, L)), transitions), axis = 0)
      

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


def first_ord_energy(v1, v2, J, N):

       des_ind = jnp.where((v1-v2) == 1, size = 1)[0]
       cre_ind = jnp.where((v1-v2) == -1, size = 1)[0]
       sgn1 = Jordan_Wigner_OP(v1, 0, des_ind[0])
       v_intermediate = v1.at[des_ind[0]].set(0)
       sgn2 = Jordan_Wigner_OP(v_intermediate, 0, cre_ind[0])
       H = sgn1*sgn2*J[cre_ind[0], des_ind[0]]

       return H
       


#@partial(jax.vmap, in_axes = (0, 0, None, None))
def energy_elements_random_hop(v, trans, J, L, N):
      H_syk = jnp.zeros((N*N)+1, dtype = complex)



      H_syk = H_syk.at[0].set(zeroth_energy_experiment(v, J, N))
      for i in range(N*N):
          H_syk = H_syk.at[i+1].set(first_ord_energy(v, trans[i+1], J, N))

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
    eta = single_trans_hop(sigma, L, N)

   
    # generate deterministically the associated entries of the syk hamiltonian
    mels = energy_elements_random_hop(sigma, eta, J, L, N)
    
    return eta, mels



L = int(sys.argv[1])
N = int(L/2)
alpha = int(sys.argv[2])
seed = int(sys.argv[3])
steps = int(sys.argv[4])
samples = int(sys.argv[5])
lr = [0.1, 0.01, 0.005, 0.001]
J = jnp.array(np.load("hop_matrix_L_"+str(L)+"seed_"+str(seed)+".npy"))



my_graph = nk.graph.Chain(length = L)
hi = Fock(n_max=1, n_particles=N, N=L)
X_OP = XOperator(hi)









#optimizer = nk.optimizer.Adam()
optimizer = nk.optimizer.Sgd(learning_rate = 0.001)

#print("For this value of the seed the exact GS energy is : ", Exact_ground(seed))
#model = Module()
model = nk.models.RBM(dtype = complex, alpha = alpha)
#model = Jastrow()
#model = nk.models.ARNNConv2D()
vs  = nk.vqs.MCState(nk.sampler.MetropolisExchange(hilbert =hi, graph = my_graph, d_max = L),model = model , n_samples = samples)
sr = nk.optimizer.SR()
gs = nk.VMC(hamiltonian = X_OP, optimizer = optimizer, variational_state=vs, preconditioner=sr)




ID = "rh_run_L_"+str(L)+"seed_"+"{:.2f}".format(seed)+"alpha_"+"{:.2f}".format(alpha)+"samples_"+str(samples)

logs = [
    nk.logging.JsonLog(ID, save_params=True, save_params_every=1),
    #nk.logging.StateLog("syk_mean", tar=True),
]

for step in range(len(lr)):
   gs.optimizer = nk.optimizer.Sgd(learning_rate = lr[step])
   gs.run(steps, out = logs)

#gs.run(steps, out=logs)





