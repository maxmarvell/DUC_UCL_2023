import numpy as np
from scipy.linalg import expm 

P_path = "./Sample_Perturbation_2.csv"
G = np.loadtxt(P_path,delimiter=',',dtype='complex_')

print(np.linalg.norm(G))

G /= np.linalg.norm(G)

print(np.linalg.norm(1e-7*G))