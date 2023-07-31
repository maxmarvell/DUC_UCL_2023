import numpy as np
from scipy.optimize import minimize
from time import time
import random
import sys

def main():

    seed_value = random.randrange(sys.maxsize)
    random.seed(seed_value)

    start = time()
    print("\n")

    q = 2
    res = constrained_DU(q)

    end = time()
    print(end-start)
    print("\n")

    W = np.full(shape=[q**2,q**2,q**2,q**2],fill_value=np.nan,dtype='complex_')

    W[0,0,0,0] = 1

    for i in  range(q**2):
        for j in range(q**2):
            if i != 0 or j != 0: 
                W[i,j,0,0] = 0
                W[i,0,j,0] = 0
                W[0,0,i,j] = 0
                W[0,i,0,j] = 0

    indicies = np.argwhere(np.isnan(W))

    for index, i in zip(indicies,res):
            W[*index] = i

    print(W)
    print("\n")

    norm = np.linalg.norm(W)
    II = np.einsum("ac,bd->abcd", np.identity(q**2), np.identity(q**2))
    unitarity = np.linalg.norm(np.einsum('abcd,efcd -> abef', W, np.conj(W))-II)
    dual_unitarity = np.linalg.norm(np.einsum('fbea,fdec->abcd',np.conj(W),W)-II)

    I, Z = np.zeros(q**2), np.zeros(q**2)
    I[0], Z[1] = 1, 1
    IZ = np.einsum('a,b->ab',I,Z)
    ZI = np.einsum('a,b->ab',Z,I)
    soliton = np.linalg.norm(np.einsum('abcd,cd->ab',W,ZI)-IZ) + np.linalg.norm(np.einsum('abcd,cd->ab',W,IZ)-ZI)

    print("Norm is:", norm)
    print("Normalised? (i.e. norm/q = 1?):", np.linalg.norm(W)/(q**2))
    print("Check unitarity: ", unitarity)
    print("Check dual unitarity: ", dual_unitarity)

    print("Check Z is a soliton: ",soliton)
    
    if unitarity < 1e-03 and dual_unitarity < 1e-03 and soliton < 1e-03 and abs(norm - (q**2)) < 1e-3:
        np.savetxt(f"./data/FoldedTensors/DU_{q}_{seed_value}.csv",W.reshape(q**4,q**4),delimiter=",")

def real_to_complex(z):
    return z[:len(z)//2] + 1j * z[len(z)//2:]

def complex_to_real(z): 
    return np.concatenate((np.real(z), np.imag(z)))

def constrained_DU(q:int):

    empty = q**8 - (4*q**4 - 4*q**2 + 1)

    W = np.full(shape=[q**2,q**2,q**2,q**2],fill_value=np.nan,dtype='complex_')
    R = real_to_complex(np.random.rand(2*empty))
    x0 = np.random.rand(2*empty)

    W[0,0,0,0] = 1

    for i in  range(q**2):
        for j in range(q**2):
            if i != 0 or j != 0: 
                W[i,j,0,0] = 0
                W[i,0,j,0] = 0
                W[0,0,i,j] = 0
                W[0,i,0,j] = 0

    indicies = np.argwhere(np.isnan(W))
    
    def unitarity(x:np.ndarray,W:np.ndarray,q:int,indicies:np.ndarray):

        for index, i in zip(indicies,x):
            W[*index] = i

        II = np.einsum("ac,bd->abcd",np.identity(q**2),np.identity(q**2))

        return np.linalg.norm(np.einsum('abcd,efcd -> abef', W, np.conj(W))-II)
    
    def dual_unitarity(x:np.ndarray,W:np.ndarray,q:int,indicies:np.ndarray):

        for index, i in zip(indicies,x):
            W[*index] = i

        II = np.einsum("ac,bd->abcd",np.identity(q**2),np.identity(q**2))

        return np.linalg.norm(np.einsum('fbea,fdec->abcd',np.conj(W),W)-II)
    
    def soliton(x:np.ndarray,W:np.ndarray,q:int,indicies:np.ndarray):

        for index, i in zip(indicies,x):
            W[*index] = i

        I, Z = np.zeros(q**2), np.zeros(q**2)

        I[0], Z[1] = 1, 1

        IZ = np.einsum('a,b->ab',I,Z)
        ZI = np.einsum('a,b->ab',Z,I)

        return np.linalg.norm(np.einsum('abcd,cd->ab',W,ZI)-IZ) + np.linalg.norm(np.einsum('abcd,cd->ab',W,IZ)-ZI)
    
    def randomise(X:np.ndarray, R:np.ndarray):
        return -abs(np.einsum("a,a->", np.conj(X), R))

    cons = [
            {'type':'eq', 'fun': lambda z:  unitarity(real_to_complex(z),W,q,indicies)}, 
            {'type':'eq', 'fun': lambda z: dual_unitarity(real_to_complex(z),W,q,indicies)},
            {'type':'eq', 'fun': lambda z: soliton(real_to_complex(z),W,q,indicies)}
            ]
    
    res = minimize(
        lambda z: randomise(real_to_complex(z),R),
        x0 = x0,
        method='SLSQP',
        constraints=cons,
        options={'maxiter': 1000},
        )
    
    return real_to_complex(res.x)

if __name__ == "__main__":
    for _ in range(20):
        main()

# class FoldedDUwithSoliton:

#     def __init__(self,q):
#         self.q = q
#         self.W = np.full(shape=[q**2,q**2,q**2,q**2],fill_value=np.nan)
#         self.W[0,0,0,0] = 1  

#         for i in  range(self.q**2):
#             for j in range(self.q**2):
#                 if i != 0 or j != 0: 
#                     self.W[i,j,0,0] = 0
#                     self.W[i,0,j,0] = 0
#                     self.W[0,0,i,j] = 0
#                     self.W[0,i,0,j] = 0

#     def update(self,x):


#     def constrain(self):
#         pass