from path_integral_random import get_gates, distribute
import quimb.tensor as qtn
import numpy as np
import pandas as pd
import math
import os

def main():
    pertubations = [3e-7]
    temperature = [0.5,0.9,1.1,1.5]
    generate_data(2,pertubations,9,temperature)

def generate_data(q:int,
                  pertubations:np.ndarray,
                  tspan:int,
                  temperatures:np.ndarray):
        
    for e in pertubations:
            
        for temp in temperatures:
    
            PW, gates = get_gates(q,e)

            df = pd.DataFrame()
            err = pd.Series()

            for T in range(2*tspan+1):

                t = float(T)/2

                data = np.array([])
                inds = np.array([])

                for x in range(-T,T+1):
                    x = float(x)/2
                    inds = np.append(inds,x)
                    data = np.append(data,[exact_contraction(x,t,q,PW,gates,temp)])

                s = pd.Series(data,inds,name=t)
                err[t] = np.abs(sum(data))

                df = pd.concat([df, s.to_frame().T])

            df = df.reindex(sorted(df.columns,key=lambda num: float(num)), axis=1)
            df = df.fillna(0)
            print(df,'\n')

            path = 'data/RandomTensorExact'

            try:
                df.to_csv(path+f'/{q}_{e}_{temp}.csv')
            except:
                os.mkdir(path)
                df.to_csv(path+f'/{q}_{e}_{temp}.csv')

            try:
                err.to_csv(path+f'/charge_conservation/{q}_{e}_{temp}.csv')
            except:
                os.mkdir(path+'/charge_conservation')
                err.to_csv(path+f'/charge_conservation/{q}_{e}_{temp}.csv')

def DU_network_construction(x:float,
                            t:float,
                            q:int,
                            PW:dict,
                            gates:np.ndarray,
                            temp:float,
                            draw:bool = False):
        
        Z, I = np.zeros([q**2]), np.zeros([q**2])
        Z[1], I[0] = 1, 1

        x_h, x_v = math.ceil(t+x), math.floor(t+1-x)

        DU = [[qtn.Tensor(

            PW[distribute(gates,temp)],
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

def exact_contraction(x:float,
                      t:float,
                      q:int,
                      PW:dict,
                      gates:np.ndarray,
                      temp:float):
    
    if x == 0 and t == 0:
        return 1

    TN = DU_network_construction(x,t,q,PW,gates,temp)

    return TN.contract()

if __name__ == '__main__':
    main()