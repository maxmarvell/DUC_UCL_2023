from P_generator import Exponential_Map
from time import time
import numpy as np
import pandas as pd
import re
import os
import math

def main():

    q = 2
    k = 3
    tspan = 12
    pertubations = [1e-7,3e-7,5e-7]

    start = time()

    for e in pertubations:
        generate_data(q,tspan,e,k=k)
    
    end = time()

    print('Total time taken to run:', end-start, '\n')

def generate_data(q:int,
                  tspan:int,
                  e:float,
                  k:int = np.inf,
                  depth:int = 0):

    if k != np.inf and depth:
        path = 'data/LightconeFrontTruncated'
    elif k != np.inf:
        path = 'data/PathIntegralTruncated'
    elif depth:
        path = 'data/LightconeFront'
    else:
        path = 'data/PathIntegral'
    
    rstr = r'DU_' + str(q) + r'_([0-9]*).csv'
    rx = re.compile(rstr)

    for _, _, files in os.walk('data/FoldedTensors'):
        for file in files[:1]:

            res = rx.match(file)
            seed = res.group(1)
    
            W = np.loadtxt(f'data/FoldedTensors/DU_{q}_{seed}.csv',
                            delimiter=',',dtype='complex_')

            P = np.loadtxt(f'data/FoldedPertubations/P_{q}_866021931.csv',
                           delimiter=',',dtype='complex_')
            
            G = Exponential_Map(e,P)

            PW = np.einsum('ab,bc->ac',G,W).reshape(q**2,q**2,q**2,q**2)

            I, Z = np.zeros(q**2), np.zeros(q**2)
            I[0], Z[1] = 1, 1

            IZ = np.einsum('a,b->ab',I,Z).flatten()
            ZI = np.einsum('a,b->ab',Z,I).flatten()

            Q = IZ + ZI

            Q2 = np.linalg.norm(np.einsum('ab,b->a',PW.reshape(q**4,q**4),Q) - Q)
            print('\n     Check Q-Conservation of peturbed P: ', Q2, '\n')

            df = pd.DataFrame()
            err = pd.Series()

            for T in range(2*tspan+1):

                t = float(T)/2

                data = np.array([])
                inds = np.array([])

                if depth:
                    minT = T - depth
                else:
                    minT = -T

                for x in range(minT,T+1):
                    x = float(x)/2
                    inds = np.append(inds,x)
                    data = np.append(data,[path_integral(x,t,PW,k=k)])

                s = pd.Series(data,inds,name=t)

                df = pd.concat([df, s.to_frame().T])
                err[t] = np.abs(sum(data))
                print(f'Time computed up to: {t}')
            
            df = df.reindex(sorted(df.columns,key=lambda num: float(num)), axis=1)
            df = df.fillna(0)
            if depth: df = df.iloc[:,depth:]
            df = df.iloc[::-1]
            print('\n',df,'\n')

            try:
                df.to_csv(f'{path}/{q}_{seed}_{e}.csv', index=False)
            except:
                os.mkdir(path)
                df.to_csv(f'{path}/{q}_{seed}_{e}.csv', index=False)

            try:
                err.to_csv(f'{path}/charge_conservation/QC_{q}_{seed}_{e}.csv', index=False)
            except:
                os.mkdir(f'{path}/charge_conservation')
                err.to_csv(f'{path}/charge_conservation/QC_{q}_{seed}_{e}.csv', index=False)

def path_integral(x:float,
                  t:float,
                  W:np.ndarray,
                  k:int = np.inf):

    def transfer_matrix(a:np.ndarray,
                        l:int,
                        horizontal:bool = True,
                        terminate:bool = False):

        '''
            A transfer matrix can either be horizontal or vertical row of contracted
            folded tensors. 
            For the case of a horizontal transfer matrix the row can only be terminated
            either by a defect or by the operator b

            a can be like [0,1,0,0]
        '''

        if horizontal:
            direct = W[0,:,:,0]
            defect = W[:,0,:,0]
        else:
            direct = W[:,0,0,:]
            defect = W[0,:,0,:]

        if not terminate:
            for _ in range(l-1):
                a = np.einsum('ab,b->a',direct,a)
            return np.einsum('ab,b->a',defect,a)

        elif ((int(2*(t+x))%2 == 0) and horizontal) or ((int(2*(t+x))%2 != 0) and not horizontal):
            for _ in range(l):
                a = np.einsum('ab,b->a',direct,a)
            return a[1]

        else:
            for _ in range(l-1):
                a = np.einsum('ab,b->a',direct,a)
            a = np.einsum('ab,b->a',defect,a)
            return a[1]

    def skeleton(h:np.ndarray,
                 v:np.ndarray,
                 a:np.ndarray):
        '''
            Computes a contribution to the path integral of the
            input skeleton diagram, only for cases where y is an integer!
        '''

        if len(v) == len(h):
            for i in range(len(v)-1):
                a = transfer_matrix(a,h[i])
                a = transfer_matrix(a,v[i],horizontal=False)
            a = transfer_matrix(a,h[-1])
            return transfer_matrix(a,v[-1],terminate=True,horizontal=False)

        else:
            for i in range(len(v)):
                a = transfer_matrix(a,h[i])
                a = transfer_matrix(a,v[i],horizontal=False)
            return transfer_matrix(a,h[-1],terminate=True) 

    a = np.array([0,1,0,0],dtype='complex_')

    if x == 0 and t == 0.:
        return a[1]

    vertical_data = {}
    horizontal_data = {}

    x_h, x_v = math.ceil(t+x), math.floor(t+1-x)

    k = min(x_h,x_v,k)

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

def list_generator(x:int,
                   data:dict,
                   k:int=np.inf,
                   lists:np.ndarray=[]):
    '''
        Generates a complete set of possible lists which can
        combine to form a complete set 
    '''

    if x == 0:
        try:
            data[len(lists)].append(lists)
        except:
            data[len(lists)] = [lists]
        return
    elif len(lists) >= k:
        return 

    for i in range(1,x+1):
        sublist = lists.copy()
        sublist.append(i)
        list_generator(x-i,data,k,sublist)

if __name__ == '__main__':
    main()