from path_integral import list_generator
from P_generator import Exponential_Map
import numpy as np
import pandas as pd
import math
import os
import re

def main():

    q = 2
    e = [1e-07]
    tspan = 10
    temp = 0.7

    for i in e:
        generate_data(q,tspan,i,temp)

def generate_data(q:int,
                  tspan:int,
                  e:float,
                  temp:float,
                  k:int = 0):
    
    PW, gates = get_gates(q,e)
    circuit = random_circuit(tspan,gates,temp)

    if k:
        path = f'data/RandomPathIntegralTruncated{temp}'
    else:
        path = f'data/RandomPathIntegral{temp}'

    df = pd.DataFrame()
    err = pd.Series()

    for T in range(2*tspan+1):

        t = float(T)/2

        data = np.array([])
        inds = np.array([])

        for x in range(-T,T+1):
            x = float(x)/2
            inds = np.append(inds,x)
            data = np.append(data,[path_integral(x,t,circuit,PW,k)])

        s = pd.Series(data,inds,name=t)
        err[t] = np.abs(sum(data))

        df = pd.concat([df, s.to_frame().T])

    df = df.reindex(sorted(df.columns,key=lambda num: float(num)), axis=1)
    df = df.fillna(0)
    print(df,'\n')

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
    
def path_integral(x:float,
                  t:float,
                  circuit:pd.DataFrame,
                  PW:dict,
                  k:int = 0):

    def transfer_matrix(a:np.ndarray,
                        l:int,
                        X:float,
                        T:float,
                        horizontal:bool = True,
                        terminate:bool = False):

        '''
            A transfer matrix can either be horizontal or vertical row of contracted
            folded tensors. 
            For the case of a horizontal transfer matrix the row can only be terminated
            either by a defect or by the operator b

            a can be like [0,1,0,0]
        '''

        if not terminate:

            # print(f'      Computing transfer matrix len {l}, Horizontal? {horizontal}')

            for _ in range(l-1):

                W = PW[circuit.loc[T,X]]

                # print('       W loaded')

                if horizontal:
                    X = X + 0.5
                    T = T + 0.5
                    direct = W[0,:,:,0]
                else:
                    X = X - 0.5
                    T = T + 0.5
                    direct = W[:,0,0,:]

                a = np.einsum('ab,b->a',direct,a)

                # print(f'       T and X updated to {T} and {X}')

            W = PW[circuit.loc[T,X]]

            # print('       W loaded')

            if horizontal:
                T = T + 0.5
                defect = W[:,0,:,0]
            else:
                T = T + 0.5
                defect = W[0,:,0,:]

            # print(f'       T and X updated to {T} and {X}')

            return np.einsum('ab,b->a',defect,a)

        elif terminate and (int(2*(t+x))%2 == 0):

            # print('    TERMINATING WITH DIRECT')

            # print(f'      Computing transfer matrix len {l}, Horizontal? {horizontal}')

            for _ in range(l):

                W = PW[circuit.loc[T,X]]

                # print('       W loaded')

                if horizontal:
                    X = X + 0.5
                    T = T + 0.5
                    direct = W[0,:,:,0]
                else:
                    X = X - 0.5
                    T = T + 0.5
                    direct = W[:,0,0,:]

                a = np.einsum('ab,b->a',direct,a)

                # print(f'       T and X updated to {T} and {X}')

            return np.einsum('a,a->',b,a)

        else:

            # print('    TERMINATING WITH DEFECT')

            # print(f'      Computing transfer matrix len {l}, Horizontal? {horizontal}')

            for _ in range(l-1):

                W = PW[circuit.loc[T,X]]

                # print('       W loaded')

                if horizontal:
                    X = X + 0.5
                    T = T + 0.5
                    direct = W[0,:,:,0]
                else:
                    X = X - 0.5
                    T = T + 0.5
                    direct = W[:,0,0,:]

                a = np.einsum('ab,b->a',direct,a)

                # print(f'       T and X updated to {T} and {X}')

            W = PW[circuit.loc[T,X]]

            # print('       W loaded')

            if horizontal:
                T = T + 0.5
                defect = W[:,0,:,0]
            else:
                T = T + 0.5
                defect = W[0,:,0,:]

            a = np.einsum('ab,b->a',defect,a)

            # print(f'       T and X updated to {T} and {X}')

            return np.einsum('a,a->',b,a)
        
    def skeleton(h:np.ndarray,
                 v:np.ndarray,
                 a:np.ndarray):
        '''
            Computes a contribution to the path integral of the
            input skeleton diagram
        '''

        X,T = 0,0

        for i in range(len(v)):

            a = transfer_matrix(a,h[i],X,T)

            X += (h[i] - 1) / 2
            T += h[i]/2

            # print(f'    GLOBAL T:{T}, X:{X}')

            a = transfer_matrix(a,v[i],X,T,horizontal=False)

            X -= (v[i] - 1) / 2
            T += v[i]/2

            # print(f'    GLOBAL T:{T}, X:{X}')

        return transfer_matrix(a,h[-1],X,T,terminate=True) if len(h) > len(v) else np.einsum('a,a->',a,b)

    a, b = np.array([0,1,0,0],dtype='complex_'), np.array([0,1,0,0],dtype='complex_')

    if x == 0 and t == 0:
        return np.abs(np.einsum('a,a->',a,b))

    vertical_data = {}
    horizontal_data = {}

    x_h, x_v = math.ceil(t+x), math.floor(t+1-x)

    if k > x_h or k > x_v or not k:
        k = min(x_h,x_v)

    list_generator(x_v-1,vertical_data,k=k)
    list_generator(x_h,horizontal_data,k=k)

    n = 1
    sum = 0

    while n <= k:

        # print(f'\nComputing skeleton diagram for {n} turns!')

        try:
            l1, l2 = horizontal_data[n], vertical_data[n]
            for h in l1:
                for v in l2:
                    # print(f'   Diagram h:{h} and v:{v}')
                    sum += skeleton(h,v,a)
        except:pass
            
        try:
            l1, l2 = horizontal_data[n+1], vertical_data[n]
            for h in l1:
                for v in l2:
                    # print(f'   Diagram h:{h} and v:{v}')
                    sum += skeleton(h,v,a)
        except:pass
                        
        try:
            l1, v = horizontal_data[n], vertical_data[0]
            for h in l1:
                # print(f'   Diagram h:{h} and v:{v}')
                sum += skeleton(h,[],a)
        except:pass

        n += 1

    return sum

def random_circuit(tspan:int,
                   gates:np.ndarray,
                   temp:float):
    
    circuit = pd.DataFrame()

    for T in range(2*tspan):

        t = float(T)/2

        row = np.array([])
        inds = np.array([])

        for x in range(-T,T+1,2):
            x = float(x)/2
            inds = np.append(inds,x)
            inds = np.append(inds,x+0.5)
            gate = distribute(gates,temp)
            row = np.append(row,gate)
            row = np.append(row,gate)

        floquet = pd.Series(row,inds,name=t)

        circuit = pd.concat([circuit, floquet.to_frame().T])

    return circuit

def distribute(gates:np.ndarray,
               temp:float):

    def thermal(x:float):
        return np.exp(-x/temp)
    
    x = np.linspace(0,1,len(gates)+1)

    p = thermal(x[:-1]) - thermal(x[1:])

    p /= np.sum(p)

    return np.random.choice(gates,p=p)
        
def get_gates(q:int,
              e:float,
              pure:bool = False):

    PW, gates = dict(), list()

    rstr = f'DU_{q}' + r'_([0-9]*).csv'
    rx = re.compile(rstr)

    for _, _, files in os.walk('data/FoldedTensors'):
        for file in files:

            res = rx.match(file)
            seed = int(res.group(1))

            gates.append(seed)
    
            W = np.loadtxt(f'data/FoldedTensors/DU_{q}_{seed}.csv',
                           delimiter=',',dtype='complex_')
            
            if pure: 
                PW[seed] = W.reshape(q**2,q**2,q**2,q**2)
                continue

            P = np.loadtxt(f'data/FoldedPertubations/P_{q}_866021931.csv',
                           delimiter=',',dtype='complex_')
            
            G = Exponential_Map(e,P)

            PW[seed] = np.einsum('ab,bc->ac',G,W).reshape(q**2,q**2,q**2,q**2)
            
    return PW, gates

if __name__ == '__main__':
    main()