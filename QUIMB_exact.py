from P_generator import Exponential_Map
from time import time
import quimb.tensor as qtn
import pandas as pd
import numpy as np
import math
import re
import os

'''
    Code to compute benchmark exact correlations for DU circuit
    depth t and second point operator loaction x
'''

def main():

    q = 2
    timespan = 50
    pertubations = [9e-8,1e-7,3e-7,5e-7]

    for e in pertubations:
        generate_data(q,timespan,e)

def generate_data(q:int,
                  tspan:int,
                  e:float):

    rstr = f'DU_{q}_' + r'([0-9]*).csv'
    rx = re.compile(rstr)

    for _, _, files in os.walk('data/FoldedTensors'):
        for file in files[:1]:

            res = rx.match(file)

            if not res:
                continue

            seed = res.group(1)
    
            W = np.loadtxt(f'./data/FoldedTensors/DU_{q}_{seed}.csv',
                            delimiter=',',dtype='complex_')

            G = np.loadtxt(f'data/FoldedPertubations/P_{q}_866021931.csv',
                           delimiter=',',dtype='complex_')

            P = Exponential_Map(e,G)
            PW = np.einsum('ab,bc->ac',P,W).reshape(q**2,q**2,q**2,q**2)

            df = pd.DataFrame()
            err = pd.Series()

            T = 0
            start = time()
            end = time()

            while end-start < tspan:

                t = float(T)/2

                if t == 0:
                    s = pd.Series(np.array([1],dtype='complex_'),np.array([0]),name=t)        
                    df = pd.concat([df, s.to_frame().T])
                    err[t] = 1
                    end = time()
                    T += 1
                    print(f'computed for T = {t}s')
                    continue

                data = np.array([],dtype='complex_')
                inds = np.array([])

                for x in range(-T+1,T+1):
                    x = float(x) / 2
                    inds = np.append(inds,x)
                    data = np.append(data,exact_contraction(x,t,q,PW))

                s = pd.Series(data,inds,name=t)
            
                df = pd.concat([df, s.to_frame().T])
                print(f'\nTime computed up to: {t}')

                err[t] = np.abs(sum(data))

                T += 1
                end = time()
            

            df = df.reindex(sorted(df.columns,key=lambda num: float(num)), axis=1)
            df = df.fillna(0)
            df = df.iloc[::-1]
            print('\n',df,'\n')

            print('Time taken to compute light cone with complete contraction: ', end-start)

            path = 'data/TensorExact'

            try:
                df.to_csv(f'{path}/{q}_{seed}_{e}.csv',
                          index=False,header=False)
            except:
                os.mkdir(path)
                df.to_csv(f'{path}/{q}_{seed}_{e}.csv',
                          index=False,header=False)

            try:
                err.to_csv(f'{path}/charge_conservation/QC_{q}_{seed}_{e}.csv',
                           index=False,header=False)
            except:
                os.mkdir(f'{path}/charge_conservation')
                err.to_csv(f'{path}/charge_conservation/QC_{q}_{seed}_{e}.csv',
                           index=False,header=False)

def exact_contraction(x:float,
                      t:float,
                      q:int,
                      W:np.ndarray,
                      draw:bool = False):
    
    if x == 0 and t == 0:
        return 1
    
    Z, I = np.zeros([q**2]), np.zeros([q**2])
    Z[1], I[0] = 1, 1

    x_h, x_v = math.ceil(t+x), math.floor(t+1-x)

    DU = [[qtn.Tensor(

        W.reshape([q**2,q**2,q**2,q**2]),
        inds=(f'k{2*i+1},{2*j+2}',f'k{2*i+2},{2*j+1}',
                f'k{2*i},{2*j+1}',f'k{2*i+1},{2*j}'),
        tags=[f'UNITARY_{i},{j}'])

        for i in range(x_h)] for j in range(x_v)]

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

    TN = qtn.TensorNetwork((DU,a,b,e1,e2,e3,e4))

    if draw:

        fix = {
            'UNITARY_0,0': (0, 0),
            f'UNITARY_0,{x_v-1}': (0, 1),
            f'UNITARY_{x_h-1},0': (1, 0),
            f'UNITARY_{x_h-1},{x_v-1}': (1, 1),
        }

        if x_h > 1 and x_v > 1:
            TN.draw(fix=fix)
        else:
            TN.draw()

    return TN.contract()

if __name__ == '__main__':
    main()