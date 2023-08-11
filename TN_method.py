from time import time
import quimb.tensor as qtn
import numpy as np
import pandas as pd
import math
import re
import os

'''
    Code to compute benchmark exact correlations for DU circuit
    depth t and second point operator loaction x
'''

def main():

    start = time()

    q = 2

    rstr = r'DU_' + str(q) + r'_([0-9]*).csv'
    rx = re.compile(rstr)

    for _, _, files in os.walk('data/FoldedTensors'):
        for file in files[:]:
            res = rx.match(file)
            seed_value = res.group(1)
    
            W = np.loadtxt(f'./data/FoldedTensors/DU_{q}_{seed_value}.csv',
                            delimiter=',',dtype='complex_')
            
            for i in range(2,10):

                df = pd.DataFrame()

                e = np.round(i,1)*(10**-8)

                P = np.loadtxt(f'./data/FoldedPertubations/P_{q}_{i}e-08_{seed_value}.csv',
                                delimiter=',',dtype='complex_')

                pW = np.einsum('ab,bc->ac',P,W)

                for T in range(2*10):

                    t = float(T)/2

                    data = np.array([],dtype='complex_')

                    for x in range(-T+1,T+1):

                        x = float(x) / 2
                        data = np.append(data,exact_contraction(x,t,q,pW))

                    s = pd.Series(data,range(-T+1,T+1),name=t)

                    df = pd.concat([df, s.to_frame().T])
                    
                print(f'For a pertubation e:{e}-08 the greatest deviation in charge conservation is: ', sum(data))

                print(df)

                try:
                    df.to_csv(f'./data/QUIMB_exact/heatmap_{q}_{i}e-08_{seed_value}.csv', index=False)
                except:
                    os.mkdir('./data/QUIMB_exact/')
                    df.to_csv(f'./data/QUIMB_exact/heatmap_{q}_{i}e-08_{seed_value}.csv', index=False)

    end = time()

    print('\nTime taken to run:', end-start)

def exact_contraction(x:float,
                      t:float,
                      q:int,
                      W:np.ndarray):

    Z, I = np.zeros([q**2]), np.zeros([q**2])
    Z[1], I[0] = 1, 1

    x_h, x_v = math.ceil(t+x), math.floor(t+1-x)

    DU = [[qtn.Tensor(W.reshape([q**2,q**2,q**2,q**2]),inds=('a','b','c','d'),
                        tags=[f'UNITARY_{i},{j}']) for i in range(x_h)] for j in range(x_v)]

    a = [qtn.Tensor(Z,inds=('k0,1',),tags=['Z'])]
    
    if int(2*(t+x))%2 == 0:
        b = [qtn.Tensor(Z,inds=(f'k{2*x_h},{2*x_v-1}',),tags=['Z'])]
        e3 = [qtn.Tensor(I,inds=(f'k{2*x_h},{2*j+1}',),tags=['I']) for j in range(x_v-1)]
        e4 = [qtn.Tensor(I,inds=(f'k{2*j+1},{2*x_v}',),tags=['I']) for j in range(x_h)]
    else:
        b = [qtn.Tensor(Z,inds=(f'k{2*x_h-1},{2*x_v}',),tags=['Z'])]
        e3 = [qtn.Tensor(I,inds=(f'k{2*x_h},{2*j+1}',),tags=['I']) for j in range(x_v)]
        e4 = [qtn.Tensor(I,inds=(f'k{2*j+1},{2*x_v}',),tags=['I']) for j in range(x_h-1)]

    e1 = [qtn.Tensor(I,inds=(f'k{2*j+1},0',),tags=['I']) for j in range(x_h)]
    e2 = [qtn.Tensor(I,inds=(f'k0,{2*j+1}',),tags=['I']) for j in range(1,x_v)]

    for i in range(x_h):
        for j in range(x_v):
            index_map = {'a':f'k{2*i+1},{2*j}',
                         'b':f'k{2*i},{2*j+1}',
                         'c':f'k{2*i+2},{2*j+1}',
                         'd':f'k{2*i+1},{2*j+2}'}
            DU[j][i].reindex(index_map,inplace=True)

    TN = qtn.TensorNetwork((DU,a,b,e1,e2,e3,e4))

    # fix = {
    # 'UNITARY_0,0': (0, 0),
    # f'UNITARY_0,{x_v-1}': (0, 1),
    # f'UNITARY_{x_h-1},0': (1, 0),
    # f'UNITARY_{x_h-1},{x_v-1}': (1, 1),
    # }

    # if x_h > 1 and x_v > 1:
    #     TN.draw(fix=fix)

    return TN.contract()

if __name__ == '__main__':
    main()