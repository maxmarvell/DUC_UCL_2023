import quimb as qu
import quimb.tensor as qtn
import numpy as np
from time import time
import random
import math
import re
import os
import pandas as pd

"""
    Code to compute benchmark exact correlations for DU circuit
    depth t and second point operator loaction x
"""

def main():

    start = time()

    q = 2

    rstr = r'DU_' + str(q) + r'_([0-9]*).csv'
    rx = re.compile(rstr)

    for _, _, files in os.walk("data/FoldedTensors"):
        for file in files[:1]:
            res = rx.match(file)
            seed_value = res.group(1)
            random.seed(seed_value)
    
            W = np.loadtxt(f'./data/FoldedTensors/DU_{q}_{seed_value}.csv',
                            delimiter=',',dtype='complex_')
            
            for e in np.linspace(0.01,0.1,11):

                df = pd.DataFrame()

                e = str(np.round(e,4)).ljust(5,'0')

                P = np.loadtxt(f'./data/FoldedPertubations/P_{q}_' + e + f'_{seed_value}.csv',
                                delimiter=',',dtype='complex_')

                pW = np.einsum('ab,bc->ac',P,W)

                for t in range(10):

                    data = []

                    for x in range(t+1):

                        data.append(exact_contraction(x,t,q,pW))

                    s = pd.Series(data,range(t+1))

                    df = pd.concat([df, s.to_frame().T])

                    print(df)

                df.to_csv(f"./data/TnMethod/heatmap_{q}_" + e + f"_{seed_value}.csv", index=False)

    end = time()

    print('\nTime taken to run:', end-start)

def exact_contraction(x:float,
                      t:int,
                      q:int,
                      W:np.ndarray):

    Z, I = np.zeros([q**2]), np.zeros([q**2])
    Z[1], I[0] = 1, 1

    x_h, x_v = t + math.ceil(x), t + 1 - math.ceil(x)

    DU = [[qtn.Tensor(W.reshape([q**2,q**2,q**2,q**2]),inds=('a','b','c','d'),
                        tags=[f'UNITARY_{i},{j}']) for i in range(x_h)] for j in range(x_v)]

    a, b = [qtn.Tensor(Z,inds=('k0,1',),tags=['Z'])],[qtn.Tensor(Z,inds=(f'k{2*x_h},{2*x_v-1}',),tags=['Z'])]

    e1 = [qtn.Tensor(I,inds=(f'k{2*j+1},0',),tags=['I']) for j in range(x_h)]
    e2 = [qtn.Tensor(I,inds=(f'k0,{2*j+1}',),tags=['I']) for j in range(1,x_v)]
    e3 = [qtn.Tensor(I,inds=(f'k{2*x_h},{2*j+1}',),tags=['I']) for j in range(x_v-1)]
    e4 = [qtn.Tensor(I,inds=(f'k{2*j+1},{2*x_v}',),tags=['I']) for j in range(x_h)]

    for i in range(x_h):
        for j in range(x_v):
            index_map = {'a':f'k{2*i+1},{2*j}',
                         'b':f'k{2*i},{2*j+1}',
                         'c':f'k{2*i+2},{2*j+1}',
                         'd':f'k{2*i+1},{2*j+2}'}
            DU[j][i].reindex(index_map,inplace=True)

    TN = qtn.TensorNetwork((DU,a,b,e1,e2,e3,e4))

    fix = {
    'UNITARY_0,0': (0, 0),
    f'UNITARY_0,{x_v-1}': (0, 1),
    f'UNITARY_{x_h-1},0': (1, 0),
    f'UNITARY_{x_h-1},{x_v-1}': (1, 1),
    }

    return np.abs(TN.contract())

if __name__ == "__main__":
    main()