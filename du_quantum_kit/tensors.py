from scipy.optimize import minimize
from scipy.linalg import expm 
import numpy as np
import random
from time import time
import os

def randomise(P, R):
    return -abs(np.einsum("a,a->", P, R))

def exp_map(e, P):
    return expm(e*P)

class FoldedGate(object):
    
    def __init__(self, q:int):
        self.seed = random.randrange(2**32 - 1)
        self.q = q
        self.W = self.dual_unitary()
        self.P = self.pertubation()

    def dual_unitary(self):

        np.random.seed(self.seed)

        start = time()

        q = self.q

        I, Z = np.zeros(q**2), np.zeros(q**2)
        I[0], Z[1] = 1, 1

        IZ = np.einsum('a,b->ab',I,Z)
        ZI = np.einsum('a,b->ab',Z,I)
        II = np.einsum("ac,bd->abcd", np.identity(q**2), np.identity(q**2))

        def initialise(q:int):
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

        def repack(indices,X,W):
            # NEED TO UPDATE
            for i in range(len(X)):
                index = indices[i]
                e = X[i]
                W[index[0],index[1],index[2],index[3]] = e

        def unitarity(indices,X,W):
            repack(indices,X,W)
            return np.linalg.norm(np.einsum('abcd,efcd -> abef', W, W) - II)

        def dual_unitarity(indices,X,W):
            repack(indices,X,W)
            return np.linalg.norm(np.einsum('fbea,fdec->abcd', W, W)-II)

        def soliton(indices,X,W):
            repack(indices,X,W)
            return np.linalg.norm(np.einsum('abcd,cd->ab',W,ZI)-IZ) + np.linalg.norm(np.einsum('abcd,cd->ab',W,IZ)-ZI)
        
        n_unfixed = q**8 - (4*q**4 - 4*q**2 + 1)
        X_0 = np.random.rand(n_unfixed)
        R = np.random.rand(n_unfixed)

        W = initialise(q)
        indices = np.argwhere(np.isnan(W))

        cons = [{'type':'eq', 'fun': lambda z:  unitarity(indices, z, W)},
                {'type':'eq', 'fun': lambda z: dual_unitarity(indices, z, W)},
                {'type':'eq', 'fun': lambda z: soliton(indices, z, W)}]

        result = minimize(
            lambda z: randomise(z, R), x0 = X_0, method='SLSQP', \
                constraints=cons, options={'maxiter': 1000}
                )
        
        end = time()

        print(f'\nTime taken to generate W: {end-start}s')

        repack(indices,result.x,W)
        
        N = np.linalg.norm(W)/(q**2)
        print('    Checking W Normalised:', N)

        U = np.linalg.norm(np.einsum('abcd,efcd -> abef', W, np.conj(W))-II)
        print('    Checking W unitarity: ', U)

        DU = np.linalg.norm(np.einsum('fbea,fdec->abcd',np.conj(W),W)-II)
        print('    Checking W dual unitarity: ', DU)

        S = np.linalg.norm(np.einsum('abcd,cd->ab',W,ZI)-IZ) + np.linalg.norm(np.einsum('abcd,cd->ab',W,IZ)-ZI)
        print('    Checking W has Z as a soliton: ', S)

        if U < 1e-4 and DU < 1e-4 and S < 1e-4 and abs(N - 1) < 1e-4:      
            return W.reshape(self.q**4,self.q**4)
        
        raise Exception('Failed to Converge!')

    def pertubation(self):

        np.random.seed(self.seed)

        start = time()

        q = self.q

        R = np.random.rand(q**8 - 2*q**4 + 1)
        P_0 = np.random.rand(q**8 - 2*q**4 + 1).reshape([q**4 - 1, q**4 - 1])
        P_0 = ((P_0 - P_0.T)/2).flatten()
        I, Z = np.zeros(q**2), np.zeros(q**2)
        I[0], Z[1] = 1, 1
        IZ = np.einsum('a,b->ab',I,Z).flatten()[1:]
        ZI = np.einsum('a,b->ab',Z,I).flatten()[1:]

        def antisymmetry(P):
            P = P.reshape([q**4 - 1,q**4 - 1])
            return np.linalg.norm(P + P.T)
        
        def charge_conservation(P):
            P = P.reshape([q**4 - 1,q**4 - 1])
            Q = IZ + ZI
            return np.linalg.norm(np.einsum('ab,b->a',P,Q))

        cons = [{'type':'eq', 'fun': lambda z:  antisymmetry(z)},
                {'type':'eq', 'fun': lambda z: charge_conservation(z)}]

        result = minimize(
            lambda z: randomise(z, R), x0 = P_0, method='SLSQP', \
            constraints=cons, options={'maxiter': 1000}
                        )

        result = result.x.reshape([q**4 - 1,q**4 - 1])
        P = np.zeros([np.shape(result)[0]+1, np.shape(result)[1]+1])
        P[1:,1:] = result

        end = time()

        print(f'\nTime taken to generate pertubation: {end-start}s\n')


        II = np.einsum('a,b->ab',I,I).flatten()
        IZ = np.einsum('a,b->ab',I,Z).flatten()
        ZI = np.einsum('a,b->ab',Z,I).flatten()

        Q = IZ + ZI

        AS = np.linalg.norm(P+P.T)
        print('     Check Antisymmetry: ', AS)

        U1 = np.linalg.norm(np.einsum('ab,b->a',P,II))
        print('     Check Unitality of P: ', U1)

        Q1 = np.linalg.norm(np.einsum('ab,b->a',P,Q))
        print('     Check Q-Conservation of P: ', Q1)

        e = 0.1
        G = exp_map(e,P)

        U2 = np.linalg.norm(np.einsum('ab,ac->bc', G, np.conj(G)) - np.eye(q**4))
        print('     Check Unitarity of G: ', U2)

        U3 = np.linalg.norm(np.einsum('ab,b->a',G,II) - II)
        print('     Check Unitality of G: ', U3)

        Q2 = np.linalg.norm(np.einsum('ab,b->a',G,Q) - Q)
        print('     Check Q-Conservation of G: ', Q2, '\n')
            
        return P.reshape(self.q**4,self.q**4)
    
    def save(self, dir:str='.'):

        if not os.path.isdir(dir):
            os.mkdir(dir)
            os.mkdir(dir+'/dualunitaries')
            os.mkdir(dir+'/pertubations')
        
        np.savetxt(dir+f'/dualunitaries/{self.q}_{self.seed}.csv',self.W,delimiter=',')
        np.savetxt(dir+f'/pertubations/{self.q}_{self.seed}.csv',self.P,delimiter=',')
    
