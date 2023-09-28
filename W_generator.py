import numpy as np
from scipy.stats import unitary_group
from scipy.optimize import minimize
import time


def real_to_complex(z):      # real vector of length 2n -> complex of length n
    return z[:len(z)//2] + 1j * z[len(z)//2:]


def complex_to_real(z):      # complex vector of length n -> real of length 2n
    return np.concatenate((np.real(z), np.imag(z)))


def initialize_W(q:int):

    W = np.full(shape=[q**2,q**2,q**2,q**2], fill_value=np.nan)
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

    #right now this is tailored for qubits
    for i in range(len(X)):
        index = indices[i]
        e = X[i]
        W[index[0],index[1],index[2],index[3]] = e
        

def randomise(X, R):
    return -abs(np.einsum("a,a->", X, R))


def find_W_constrained(q, soliton = False, print_result = False):

    Num_unfixed = q**8 - (4*q**4 - 4*q**2 + 1)
    X_0 = np.random.rand(Num_unfixed)
    R = np.random.rand(Num_unfixed)


    def unitarity(indices,X,W):

        repack_W(indices,X,W)
        IdId = np.einsum("ac,bd->abcd", np.identity(q**2), np.identity(q**2))
        return np.linalg.norm(np.einsum('abcd,efcd -> abef', W, W) - IdId)


    def dual_unitarity(indices,X,W):
        
        repack_W(indices,X,W)
        IdId = np.einsum("ac,bd->abcd", np.identity(q**2), np.identity(q**2))
        return np.linalg.norm(np.einsum('fbea,fdec->abcd', W, W)-IdId)


    def soliton(indices,X,W):

        repack_W(indices,X,W)
        I, Z = np.zeros(q**2), np.zeros(q**2)
        I[0], Z[1] = 1, 1
        IZ = np.einsum('a,b->ab',I,Z)
        ZI = np.einsum('a,b->ab',Z,I)
        return np.linalg.norm(np.einsum('abcd,cd->ab',W,ZI)-IZ) + np.linalg.norm(np.einsum('abcd,cd->ab',W,IZ)-ZI)



    W = initialize_W(q)
    indices = np.argwhere(np.isnan(W))

    if not soliton:

        cons = [{'type':'eq', 'fun': lambda z:  unitarity(indices, z, W)},
                {'type':'eq', 'fun': lambda z: dual_unitarity(indices, z, W)}]

    else:

        cons = [{'type':'eq', 'fun': lambda z:  unitarity(indices, z, W)},
                {'type':'eq', 'fun': lambda z: dual_unitarity(indices, z, W)},
                {'type':'eq', 'fun': lambda z: soliton(indices, z, W)}]

    result = minimize(
        lambda z: randomise(z, R), x0 = X_0, method='SLSQP', \
             constraints=cons, options={'maxiter': 1000}
                     )
    
    if print_result == True:
        print(result)
        
    return result.x




start = time.time()
print("\n\n")

q=2 #the local Hilbert space dimension
W = initialize_W(q)
indices = np.argwhere(np.isnan(W))
results = find_W_constrained(q, soliton=True)
repack_W(indices, results, W)

end = time.time()
print(end-start)
print("\n")

print(W)
print("\n\n")

print("Norm is:", np.linalg.norm(W))
print("Normalised? (i.e. norm/q = 1?):", np.linalg.norm(W)/(q**2))
IdId = np.einsum("ac,bd->abcd", np.identity(q**2), np.identity(q**2))

print("Check unitarity: ", \
      np.linalg.norm(np.einsum('abcd,efcd -> abef', W, W)-IdId))

print(
      "Check dual unitarity: ", \
      np.linalg.norm(np.einsum('fbea,fdec->abcd',W,W)-IdId)
      )

I, Z = np.zeros(q**2), np.zeros(q**2)
I[0], Z[1] = 1, 1
IZ = np.einsum('a,b->ab',I,Z)
ZI = np.einsum('a,b->ab',Z,I)

print(
      "Check Z is a soliton: ", \
          np.linalg.norm(np.einsum('abcd,cd->ab',W,ZI)-IZ) + np.linalg.norm(np.einsum('abcd,cd->ab',W,IZ)-ZI)
                  )


W = W.reshape([q**4, q**4])
np.savetxt("./Sample_Tensor_6.csv", W, delimiter=",")