from path_integral_random import get_gates, random_circuit
from path_integral_tree import Tree
from path_integral import list_generator
import numpy as np
import pandas as pd
import math
import os

def main():

    e = 1e-7
    q = 2
    temp = 0.001
    tspan = 11
    PW, gates = get_gates(q,e)
    circuit = random_circuit(tspan,gates,temp)

    tree = Tree()
    df = pd.DataFrame()
    err = pd.Series()

    for T in range(2*tspan+1):
        t = float(T)/2

        data = np.array([])
        inds = np.array([])

        for x in range(-T+1,T+1):
            x = float(x)/2
            inds = np.append(inds,x)
            data = np.append(data,[path_integral(x,t,q,circuit,PW,tree)])

        print(f'computed for T = {t}s')

        s = pd.Series(data,inds,name=t)        
        df = pd.concat([df, s.to_frame().T])
        err[t] = np.abs(sum(data))

    df = df.reindex(sorted(df.columns,key=lambda num: float(num)), axis=1)
    df = df.fillna(0)

    path = 'data/RandomPathIntegralTree'

    try:
        df.to_csv(f'{path}/{q}_{e}_{temp}.csv')
    except:
        os.mkdir(path)
        df.to_csv(f'{path}/{q}_{e}_{temp}.csv')
    
    try:
        err.to_csv(f'{path}/charge_conservation/QC_{q}_{e}_{temp}.csv')
    except:
        os.mkdir(f'{path}/charge_conservation')
        err.to_csv(f'{path}/charge_conservation/QC_{q}_{e}_{temp}.csv')

def path_integral(x:float,
                  t:float,
                  q:int,
                  circuit:pd.DataFrame,
                  PW:dict,
                  tree:Tree):
    
    def transfer_matrix(a:np.ndarray,
                        l:int,
                        X:float,
                        T:float,
                        horizontal:bool = True):

        if ((int(2*(t+x))%2 == 0) and horizontal) or ((int(2*(t+x))%2 != 0) and not horizontal):  
            for _ in range(l):
                
                W = PW[circuit.loc[T,X]]

                if horizontal:
                    X = X + 0.5
                    direct = W[0,:,:,0]
                    tree.move_right()
                else:
                    X = X - 0.5
                    direct = W[:,0,0,:]
                    tree.move_left()

                T = T + 0.5

                a = np.einsum('ab,b->a',direct,a)

            return np.einsum('a,a->',b,a)

        else:
            for _ in range(l-1):

                W = PW[circuit.loc[T,X]]

                if horizontal:
                    X = X + 0.5
                    direct = W[0,:,:,0]
                    tree.move_right()
                else:
                    X = X - 0.5
                    direct = W[:,0,0,:]
                    tree.move_left()

                T = T + 0.5
                    
                a = np.einsum('ab,b->a',direct,a)

            W = PW[circuit.loc[T,X]]

            if horizontal:
                tree.move_right()
                defect = W[:,0,:,0]
            else:
                tree.move_left()
                defect = W[0,:,0,:]

            a = np.einsum('ab,b->a',defect,a)
            tree.assign_data(a)

            return np.einsum('a,a->',b,a)

    def search_tree(l:int,
                    horizontal:bool = True):
        
        if horizontal:
            for _ in range(l):
                tree.move_right()
        else:
            for _ in range(l):
                tree.move_left()

    def skeleton(h:np.ndarray,
                 v:np.ndarray,
                 a:np.ndarray):

        tree.return_root()
        X,T = 0,0

        if v == []:
            return transfer_matrix(a,h[-1],X,T)

        elif len(v) == len(h):

            for i in range(len(v)-1):

                search_tree(h[i])
                X += (h[i] - 1) / 2
                T += h[i]/2

                search_tree(v[i],horizontal=False)
                X -= (h[i] - 1) / 2
                T += v[i]/2

            search_tree(h[-1])
            X += (h[-1] - 1) / 2
            T += h[-1]/2

            a = tree.current.data

            return transfer_matrix(a,v[-1],X,T,horizontal=False)

        elif len(h) > len(v):

            for i in range(len(v)):

                search_tree(h[i])
                X += (h[i] - 1) / 2
                T += h[i]/2

                search_tree(v[i],horizontal=False)
                X -= (v[i] - 1) / 2
                T += v[i]/2

            a = tree.current.data

            return transfer_matrix(a,h[-1],X,T)

    a, b = np.zeros(q**2,dtype='complex_'), np.zeros(q**2,dtype='complex_')
    a[1], b[1] = 1, 1

    if x == 0 and t == 0.:
        return np.einsum('a,a->',a,b)
    
    vertical_data = {}
    horizontal_data = {}

    x_h, x_v = math.ceil(t+x), math.floor(t+1-x)
    
    k = min(x_h,x_v)

    list_generator(x_v-1,vertical_data,k=k)
    list_generator(x_h,horizontal_data,k=k+1)

    n = 1
    sum = 0

    while n <= k:

        try:
            l1, l2 = horizontal_data[n], vertical_data[n]
            for h in l1:
                for v in l2:
                    sum += skeleton(h,v,a)
        except:pass
            
        try:
            l1, l2 = horizontal_data[n+1], vertical_data[n]
            for h in l1:
                for v in l2:
                    sum += skeleton(h,v,a)
        except:pass
                        
        try:
            l1, v = horizontal_data[n], vertical_data[0]
            for h in l1:
                sum += skeleton(h,[],a)
        except:pass

        n += 1

    return sum

if __name__ == '__main__':
    main()