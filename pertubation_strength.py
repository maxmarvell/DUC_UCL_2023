from P_generator import Exponential_Map
import numpy as np
import re
import os

'''
    File to develop metric that evaluates
    how dual unitary breaking the pertubation
    actually is for different e.
'''

def main():
    strength_check(2,1e-9)

def strength_check(q:int,
                   e:float):

    I, Z = np.zeros(q**2), np.zeros(q**2)
    I[0], Z[1] = 1, 1

    IZ = np.einsum('a,b->ab',I,Z)
    ZI = np.einsum('a,b->ab',Z,I)

    II = np.einsum('ac,bd->abcd', np.identity(q**2), np.identity(q**2))

    rstr = f'DU_{q}' + r'_([0-9]*).csv'
    rx = re.compile(rstr)

    for _, _, files in os.walk('data/FoldedTensors'):
        for file in files:

            res = rx.match(file)
            seed = res.group(1)

            W = np.loadtxt(f'data/FoldedTensors/DU_2_{seed}.csv',delimiter=',',dtype='complex_')

            P = np.loadtxt(f'data/FoldedPertubations/P_2_866021931.csv',delimiter=',',dtype='complex_')

            G = Exponential_Map(e,P)

            PW = np.einsum('ab,bc->ac',G,W).reshape(q**2,q**2,q**2,q**2)

            print(f'Testing seed {seed}:')

            U = np.linalg.norm(np.einsum('abcd,efcd -> abef', PW, np.conj(PW))-II)
            print('    Checking W unitarity: ', U)

            DU = np.linalg.norm(np.einsum('fbea,fdec->abcd',np.conj(PW),PW)-II)
            print('    Checking W dual unitarity: ', DU)

            S = np.linalg.norm(np.einsum('abcd,cd->ab',PW,ZI)-IZ) + np.linalg.norm(np.einsum('abcd,cd->ab',PW,IZ)-ZI)
            print('    Checking W has Z as a soliton: ', S, '\n')

if __name__ == '__main__':  
    main()