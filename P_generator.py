from scipy.optimize import minimize
from utils import *
import numpy as np

def find_P_constrained(q:int):

    R = np.random.rand(q**8 - 2*q**4 + 1)
    P_0 = np.random.rand(q**8 - 2*q**4 + 1).reshape([q**4 - 1, q**4 - 1])
    P_0 = ((P_0 - P_0.T)/2).flatten()
    I, Z = np.zeros(q**2), np.zeros(q**2)
    I[0], Z[1] = 1, 1
    IZ = np.einsum('a,b->ab',I,Z).flatten()[1:]
    ZI = np.einsum('a,b->ab',Z,I).flatten()[1:]

    def Antisymmetry(P):
        P = P.reshape([q**4 - 1,q**4 - 1])
        return np.linalg.norm(P + P.T)
    
    def Charge_conservation(P):
        P = P.reshape([q**4 - 1,q**4 - 1])
        Q = IZ + ZI
        return np.linalg.norm(np.einsum('ab,b->a',P,Q))

    cons = [{'type':'eq', 'fun': lambda z:  Antisymmetry(z)},
            {'type':'eq', 'fun': lambda z: Charge_conservation(z)}]

    result = minimize(
        lambda z: randomise(z, R), x0 = P_0, method='SLSQP', \
        constraints=cons, options={'maxiter': 1000}
                     )

    P = result.x.reshape([q**4 - 1,q**4 - 1])
    Direct_Sum = np.zeros([np.shape(P)[0]+1, np.shape(P)[1]+1])
    Direct_Sum[1:,1:] = P
        
    return Direct_Sum