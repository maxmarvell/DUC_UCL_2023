from scipy.optimize import minimize
from time import time
import numpy as np
import random
import sys
import os

'''
    Optimisation routine that generates random dual unitary folded tensors
    for dimension q.
    Additional Soliton condition enforced.
    Run to generate a seeded random value dual unitary which is saved as CSV
'''

def main():
    generate_data(2)
    
def generate_data(q:int):

    seed_value = random.randrange(2**32 - 1)
    random.seed(seed_value)
    np.random.seed(seed_value)

    start = time()
    print("\n")

    W = initialize_W(q)
    indices = np.argwhere(np.isnan(W))
    results = find_W_constrained(q)
    repack_W(indices,results,W)

    end = time()
    print(end-start)
    print("\n")

    print(W)
    print("\n")

    II = np.einsum("ac,bd->abcd", np.identity(q**2), np.identity(q**2))
    I, Z = np.zeros(q**2), np.zeros(q**2)
    I[0], Z[1] = 1, 1
    IZ = np.einsum('a,b->ab',I,Z)
    ZI = np.einsum('a,b->ab',Z,I)
    
    N = np.linalg.norm(W)/(q**2)
    print("Normalised:", np.linalg.norm(W)/(q**2))

    U = np.linalg.norm(np.einsum('abcd,efcd -> abef', W, np.conj(W))-II)
    print("Check unitarity: ", U)

    DU = np.linalg.norm(np.einsum('fbea,fdec->abcd',np.conj(W),W)-II)
    print("Check dual unitarity: ", DU)

    S = np.linalg.norm(np.einsum('abcd,cd->ab',W,ZI)-IZ) + np.linalg.norm(np.einsum('abcd,cd->ab',W,IZ)-ZI)
    print("Check Z is a soliton: ", S)
    
    if U < 1e-3 and DU < 1e-3 and S < 1e-3 and abs(N - 1) < 1e-3:
        
        try:
            np.savetxt(f"./data/FoldedTensors/DU_{q}_{seed_value}.csv",W.reshape(q**4,q**4),delimiter=",")
        except:
            os.mkdir("./data/FoldedTensors/")
            np.savetxt(f"./data/FoldedTensors/DU_{q}_{seed_value}.csv",W.reshape(q**4,q**4),delimiter=",")

def real_to_complex(z):
    return z[:len(z)//2] + 1j * z[len(z)//2:]

def complex_to_real(z):
    return np.concatenate((np.real(z), np.imag(z)))

def initialize_W(q:int):

    W = np.full(shape=[q**2,q**2,q**2,q**2], fill_value=np.nan, dtype="complex_")
    W[0,0,0,0] = 1

    for i in  range(q**2):
        for j in range(q**2):
            if i != 0 or j != 0: 
                W[i,j,0,0] = 0
                W[i,0,j,0] = 0
                W[0,0,i,j] = 0
                W[0,i,0,j] = 0
    return W 

def repack_W(indices,X,W):

    if sys.version < '3.11':
        for i in range(len(X)):
            index = indices[i]
            e = X[i]
            W[index[0],index[1],index[2],index[3]] = e

    else:
        for index, i in zip(indices,X):
                W[*index] = i
        
def randomise(X, R):
    return -abs(np.einsum("a,a->", np.conj(X), R))

def find_W_constrained(q):

    unfixed = q**8 - (4*q**4 - 4*q**2 + 1)
    X_0 = np.random.rand(2*unfixed)
    R = real_to_complex(np.random.rand(2*unfixed))

    def unitarity(indices,X,W):

        repack_W(indices,X,W)
        II = np.einsum("ac,bd->abcd", np.identity(q**2), np.identity(q**2))
        return np.linalg.norm(np.einsum('abcd,efcd -> abef', W, np.conj(W)) - II)

    def dual_unitarity(indices,X,W):

        repack_W(indices,X,W)
        II = np.einsum("ac,bd->abcd", np.identity(q**2), np.identity(q**2))
        return np.linalg.norm(np.einsum('fbea,fdec->abcd', np.conj(W), W)-II)

    def soliton(indices,X,W):

        repack_W(indices,X,W)
        I, Z = np.zeros(q**2), np.zeros(q**2)
        I[0], Z[1] = 1, 1
        IZ = np.einsum('a,b->ab',I,Z)
        ZI = np.einsum('a,b->ab',Z,I)
        return np.linalg.norm(np.einsum('abcd,cd->ab',W,ZI)-IZ) + np.linalg.norm(np.einsum('abcd,cd->ab',W,IZ)-ZI)

    W = initialize_W(q)
    indices = np.argwhere(np.isnan(W))

    cons = [{'type':'eq', 'fun': lambda z:  unitarity(indices, real_to_complex(z), W)},
            {'type':'eq', 'fun': lambda z: dual_unitarity(indices, real_to_complex(z), W)},
            {'type':'eq', 'fun': lambda z: soliton(indices, real_to_complex(z), W)}]

    result = minimize(
        lambda z: randomise(real_to_complex(z), R), x0 = X_0, method='SLSQP', \
             constraints=cons, options={'maxiter': 1000}
                     )
        
    return real_to_complex(result.x)

if __name__ == "__main__":
    for _ in range(5):
        main()