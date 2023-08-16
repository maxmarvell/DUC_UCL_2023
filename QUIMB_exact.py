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

    start = time()

    generate_data(q,8)

    end = time()

    print('\nTotal time taken to run:', end-start)

def generate_data(q:int,
                  tspan:int):

    rstr = f'DU_{q}_' + r'([0-9]*).csv'
    rx = re.compile(rstr)

    for _, _, files in os.walk('data/FoldedTensors'):
        for file in files[:1]:

            res = rx.match(file)
            seed = res.group(1)
    
            W = np.loadtxt(f'./data/FoldedTensors/DU_{q}_{seed}.csv',
                            delimiter=',',dtype='complex_')
            
            rstr2 = f'P_{q}_' + r'([0-9e\-.]*).csv'
            rx2 = re.compile(rstr2)
            
            for _, _, files in os.walk('data/FoldedPertubations'):
                for file in files:

                    res2 = rx2.match(file)
                    e = res2.group(1)

                    if not res2:
                        continue

                    start = time()

                    df = pd.DataFrame()

                    P = np.loadtxt(f'./data/FoldedPertubations/P_{q}_{e}.csv',
                                    delimiter=',',dtype='complex_')

                    pW = np.einsum('ab,bc->ac',P,W)

                    err = pd.Series()

                    for T in range(2*tspan+1):

                        t = float(T)/2

                        data = np.array([],dtype='complex_')

                        for x in range(-T,T+1):

                            x = float(x) / 2
                            data = np.append(data,exact_contraction(x,t,q,pW))

                        s = pd.Series(data,range(-T,T+1),name=t)

                        df = pd.concat([df, s.to_frame().T])

                        err[t] = np.abs(sum(data))

                        print('Light cone slice computed for t = ', t, '\n')

                    print(df)

                    end = time()

                    print('Time taken to compute light cone: ', end-start)

                    try:
                        df.to_csv(f'data/TensorExact/{q}_{seed}_{e}.csv', index=False)
                        err.to_csv(f'data/TensorExact/charge_conservation/{q}_{seed}_{e}.csv', index=False)
                    except:
                        os.mkdir('data/TensorExact')
                        os.mkdir('data/TensorExact/charge_conservation')
                        df.to_csv(f'data/TensorExact/{q}_{seed}_{e}.csv', index=False)
                        err.to_csv(f'data/TensorExact/charge_conservation/{q}_{seed}_{e}.csv', index=False)

def exact_contraction(x:float,
                      t:float,
                      q:int,
                      W:np.ndarray):
    
    if x == 0 and t == 0:
        return 1

    TN = DU_network_construction(x,t,q,W)

    return TN.contract()

def MPS_state(x:float,
              t:float,
              TN:qtn.TensorNetwork):

    x_h, x_v = math.ceil(t+x), math.floor(t+1-x)

    if x_h == 1 or x_v == 1:
        return

    for i in range(x_h):
        tags = [f'UNITARY_{i},{j}' for j in range(x_v)]
        print(tags)
        TN = TN.contract(tags=tags)

    TN.draw()

    pass

def DU_network_construction(x:float,
                            t:float,
                            q:int,
                            W:np.ndarray,
                            draw:bool = False):
        
        Z, I = np.zeros([q**2]), np.zeros([q**2])
        Z[1], I[0] = 1, 1

        x_h, x_v = math.ceil(t+x), math.floor(t+1-x)

        DU = [[qtn.Tensor(

            W.reshape([q**2,q**2,q**2,q**2]),
            inds=(f'k{2*i+1},{2*j}',f'k{2*i},{2*j+1}',
                  f'k{2*i+2},{2*j+1}',f'k{2*i+1},{2*j+2}'),
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

        return TN

if __name__ == '__main__':
    main()