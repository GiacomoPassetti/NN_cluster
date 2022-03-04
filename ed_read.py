import numpy as np
import sys

L = int(sys.argv[1])
seed = 1


print(np.load("ED_syk_energy_L"+str(L)+"seed_"+str(seed)+".npy"))