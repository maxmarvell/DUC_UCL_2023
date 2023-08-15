import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm 
from W_generator import real_to_complex, complex_to_real
import time
import os
import re

'''
    Code to generate pertubations in the folded picture.
    Note that this file creates 
'''

def main():

    q = 2
    pertubations = [1e-07,3e-07,5e-07,7e-07,9e-07]
    generate_data(q,pertubations)

def generate_data(q:int,
                  pertubations:np.ndarray):

    for _, _, files in os.walk('data/FoldedTensors'):
        for file in files[:]:

            rstr = f'DU_{q}' + r'_([0-9]*).csv'
            rx = re.compile(rstr)

            res = rx.match(file)
            seed = int(res[1])

            np.random.seed(seed)

            start = time.time()

            P = find_P_constrained(q)

            end = time.time()

            print('\nTime taken to run:',end-start)

            for e in pertubations:

                try:
                    np.loadtxt(f'data/FoldedPertubations/P_{q}_{seed}_{e}.csv',
                               delimiter=',',dtype='complex_')
                    continue
                except:
                    pass

                G = Exponential_Map(e,P)

                I, Z = np.zeros(q**2), np.zeros(q**2)
                I[0], Z[1] = 1, 1

                IZ = np.einsum('a,b->ab',I,Z).flatten()
                ZI = np.einsum('a,b->ab',Z,I).flatten()
                II = np.einsum('a,b->ab',I,I).flatten()

                Q = IZ + ZI

                H = np.linalg.norm(P - P.conj().T)
                print('Check Hermiticity: ', H)

                U1 = np.linalg.norm(np.einsum('ab,b->a',P,II))
                print('Check Unitality: ', U1)

                Q1 = np.linalg.norm(np.einsum('ab,b->a',P,Q))
                print('Check Q-Conservation: ', Q1)

                U2 = np.linalg.norm(np.einsum('ab,ac->bc', G, np.conj(G)) - np.eye(q**4))
                print('Check Unitarity: ', U2)

                U3 = np.linalg.norm(np.einsum('ab,b->a',G,II) - II)
                print('Check Unitality: ', U3)

                Q2 = np.linalg.norm(np.einsum('ab,b->a',G,Q) - Q)
                print('Check Q-Conservation: ', Q2, '\n')

                try:
                    np.savetxt(f'data/FoldedPertubations/P_{q}_{seed}_{e}.csv',G.reshape(q**4,q**4),delimiter=',')
                except:
                    os.mkdir('data/FoldedPertubations/')
                    np.savetxt(f'data/FoldedPertubations/P_{q}_{seed}_{e}.csv',G.reshape(q**4,q**4),delimiter=',')

def find_P_constrained(q:int):

    R = real_to_complex(np.random.rand(2*(q**8 - 2*q**4 + 1)))
    P_0 = real_to_complex(np.random.rand(2*(q**8 - 2*q**4 + 1))).reshape([q**4 - 1, q**4 - 1])
    P_0 = complex_to_real(((P_0 + P_0.conj().T)/2).flatten())
    
    I, Z = np.zeros(q**2), np.zeros(q**2)
    I[0], Z[1] = 1, 1
    IZ = np.einsum('a,b->ab',I,Z).flatten()[1:]
    ZI = np.einsum('a,b->ab',Z,I).flatten()[1:]

    def Hermiticity(P:np.ndarray):
        P = real_to_complex(P)
        P = P.reshape([q**4 - 1,q**4 - 1])
        return np.linalg.norm(P - P.conj().T)
    
    def Charge_conservation(P:np.ndarray):
        P = real_to_complex(P)
        P = P.reshape([q**4 - 1,q**4 - 1])
        Q = IZ + ZI
        return np.linalg.norm(np.einsum('ab,b->a',P,Q))

    def randomise(P:np.ndarray, R:np.ndarray):
        P = real_to_complex(P)
        return -abs(np.einsum('a,a->', np.conj(P), R))

    cons = [{'type':'eq', 'fun': lambda z:  Hermiticity(z)},
            {'type':'eq', 'fun': lambda z: Charge_conservation(z)}]
        
    result = minimize(
        lambda z: randomise(z, R), x0 = P_0, method='SLSQP', \
        constraints=cons, options={'maxiter': 1000}
                     )

    P = real_to_complex(result.x).reshape([q**4 - 1,q**4 - 1])
    Direct_Sum = np.zeros([np.shape(P)[0]+1, np.shape(P)[1]+1], dtype='complex_')
    Direct_Sum[1:,1:] = P

    return Direct_Sum

def Exponential_Map(e, P):
    return expm(1j*e*P)

def plot(G):

    import matplotlib.pyplot as plt

    plt.imshow(np.abs(G),cmap='hot', interpolation='nearest')
    plt.show()

if __name__ == '__main__':
    main()