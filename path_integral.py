import numpy as np
from time import time
from P_generator import Exponential_Map
import pandas as pd
import re
import os
import math

def main():

    # q, k, tspan, depth = 2, 3, 10, 6

    # pertubations = [3e-07,5e-07]

    # start = time()

    # generate_data(q,tspan,pertubations,k=k)
    
    # end = time()

    # print('Total time taken to run:', end-start, '\n')

    e = 1e-7
    q = 2
    W_path = 'data/FoldedTensors/DU_2_33296628.csv'
    P_path = 'data/FoldedPertubations/P_2_866021931.csv'
    W = np.loadtxt(W_path,delimiter=',',dtype='complex_')
    G = np.loadtxt(P_path,delimiter=',',dtype='complex_')
    P = Exponential_Map(e,G)
    PW = np.einsum('ab,bc->ac',P,W).reshape([q**2,q**2,q**2,q**2])
    tspan = 8

    df = pd.DataFrame()
    err = pd.Series()

    for T in range(2*tspan+1):
        t = float(T)/2

        data = np.array([])
        inds = np.array([])

        for x in range(-T+1,T+1):
            x = float(x)/2
            inds = np.append(inds,x)
            data = np.append(data,[path_integral(x,t,PW)])

        print(f'computed for T = {t}s')

        s = pd.Series(data,inds,name=t)        
        df = pd.concat([df, s.to_frame().T])
        err[t] = np.abs(sum(data))

    df = df.reindex(sorted(df.columns,key=lambda num: float(num)), axis=1)
    df = df.fillna(0)

    print(df)
    print(err)

def generate_data(q:int,
                  tspan:int,
                  pertubations:np.ndarray,
                  k:int = 0):

    rstr = r'DU_' + str(q) + r'_([0-9]*).csv'
    rx = re.compile(rstr)

    if k:
        path = 'data/PathIntegralTruncated'
    else:
        path = 'data/PathIntegral'

    for _, _, files in os.walk('data/FoldedTensors'):
        for file in files[:1]:

            res = rx.match(file)
            seed = res.group(1)
    
            W = np.loadtxt(f'data/FoldedTensors/DU_{q}_{seed}.csv',
                            delimiter=',',dtype='complex_')
            
            # rstr2 = f'P_{q}_' + r'([0-9e\-.]*).csv'
            # rx2 = re.compile(rstr2)

            P = np.loadtxt(f'data/FoldedPertubations/P_{q}_866021931.csv',
                           delimiter=',',dtype='complex_')
            
            for e in pertubations:

                # res2 = rx2.match(file)

                # if not res2: continue

                # e = res2.group(1)
                
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

                    for x in range(-T,T+1):
                        x = float(x)/2
                        inds = np.append(inds,x)
                        data = np.append(data,[path_integral(x,t,PW,k=k)])

                    s = pd.Series(data,inds,name=t)

                    df = pd.concat([df, s.to_frame().T])
                    err[t] = np.abs(sum(data))
                
                df = df.reindex(sorted(df.columns,key=lambda num: float(num)), axis=1)
                df = df.fillna(0)
                print(df,'\n')

                try:
                    df.to_csv(f'{path}/{q}_{seed}_{e}.csv', index=False)
                except:
                    os.mkdir(path)
                    df.to_csv(f'{path}/{q}_{seed}_{e}.csv', index=False)

                try:
                    err.to_csv(f'{path}/charge_conservation/{q}_{seed}_{e}.csv', index=False)
                except:
                    os.mkdir(f'{path}/charge_conservation')
                    err.to_csv(f'{path}/charge_conservation/{q}_{seed}_{e}.csv', index=False)

def generate_conefront(q:int,
                       tspan:int,
                       pertubations:np.ndarray,
                       depth:int,
                       k:int = 0):
    
    rstr = r'DU_' + str(q) + r'_([0-9]*).csv'
    rx = re.compile(rstr)

    if k:
        path = f'data/LightconeFrontTruncated'
    else:
        path = 'data/LightconeFront'

    for _, _, files in os.walk('data/FoldedTensors'):
        for file in files[:1]:

            res = rx.match(file)
            seed = res.group(1)
    
            W = np.loadtxt(f'data/FoldedTensors/DU_{q}_{seed}.csv',
                            delimiter=',',dtype='complex_')
            
            # rstr2 = f'P_{q}_' + r'([0-9e\-.]*).csv'
            # rx2 = re.compile(rstr2)

            P = np.loadtxt(f'data/FoldedPertubations/P_{q}_866021931.csv',                          
                           delimiter=',',dtype='complex_')
            
            for e in pertubations:

                # res2 = rx2.match(file)

                # if not res2: continue

                # e = res2.group(1)
                
                G = Exponential_Map(e,P)

                PW = np.einsum('ab,bc->ac',G,W).reshape(q**2,q**2,q**2,q**2)

                df = pd.DataFrame()

                for T in range(2*tspan+1):

                    t = float(T)/2

                    data = np.array([])
                    inds = np.array([])

                    for x in range(T-depth,T+1):
                        x = float(x)/2
                        inds = np.append(inds,x)
                        data = np.append(data,[path_integral(x,t,PW,k=k)])

                    s = pd.Series(data,inds,name=t)
                    df = pd.concat([df, s.to_frame().T])
                
                df = df.reindex(sorted(df.columns,key=lambda num: float(num)), axis=1)
                df = df.fillna(0)
                df = df.iloc[:,depth:]
                print(df,'\n')

                try:
                    df.to_csv(path+f'/{q}_{seed}_{e}.csv')
                except:
                    os.mkdir(path)
                    df.to_csv(path+f'/{q}_{seed}_{e}.csv')

def path_integral(x:float,
                  t:float,
                  W:np.ndarray,
                  k:int = 0):

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
            return np.einsum('a,a->',b,a)

        else:
            for _ in range(l-1):
                a = np.einsum('ab,b->a',direct,a)
            a = np.einsum('ab,b->a',defect,a)
            return np.einsum('a,a->',b,a)

    def skeleton(h:np.ndarray,
                 v:np.ndarray,
                 a:np.ndarray):
        '''
            Computes a contribution to the path integral of the
            input skeleton diagram, only for cases where y is an integer!
        '''

        if v == []:
            return transfer_matrix(a,h[-1],terminate=True)
        
        elif len(v) == len(h):
            for i in range(len(v)-1):
                a = transfer_matrix(a,h[i])
                a = transfer_matrix(a,v[i],horizontal=False)
            a = transfer_matrix(a,h[-1])

        else:
            for i in range(len(v)):
                a = transfer_matrix(a,h[i])
                a = transfer_matrix(a,v[i],horizontal=False)

        return transfer_matrix(a,h[-1],terminate=True) if len(h) > len(v) else transfer_matrix(a,v[-1],terminate=True,horizontal=False)

    a, b = np.array([0,1,0,0],dtype='complex_'), np.array([0,1,0,0],dtype='complex_')

    if x == 0 and t == 0.:
        return np.einsum('a,a->',a,b)

    vertical_data = {}
    horizontal_data = {}

    x_h, x_v = math.ceil(t+x), math.floor(t+1-x)

    if not k:
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

def list_generator(x:int,data:dict,k:int=np.inf,
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