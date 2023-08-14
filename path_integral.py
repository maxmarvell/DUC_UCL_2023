from time import time
import pandas as pd
import numpy as np
import random
import sys
import math
import re
import os

'''
    Computing the correlation function only where opertor a is initially localised to x 
    , where x is an integer, and operator b is localised to some point y at time 2t, 
    where y is an integer.

    TODO: '1. maybe implement random walk here

    LATER: 1. Possible SVD for larger transfer matricies??
'''

def main():

    start = time()

    q, k, tspan, depth = 2, 3, 20, 6

    generate_conefront(q,tspan,depth,k=k,truncate=True)
    
    end = time()

    print('\nTime taken to run:', end-start)

def generate_data(q:int,
                  tspan:int,
                  k:int = 0,
                  truncate:bool = False):

    rstr = r'DU_' + str(q) + r'_([0-9]*).csv'
    rx = re.compile(rstr)

    if truncate:
        path = f"data/truncated_path_k{k}/"
    else:
        path = "data/complete_path/"

    for _, _, files in os.walk("data/FoldedTensors"):
        for file in files[:]:

            res = rx.match(file)
            seed_value = res.group(1)

            err = pd.Series()
    
            W = np.loadtxt(f'./data/FoldedTensors/DU_{q}_{seed_value}.csv',
                            delimiter=',',dtype='complex_')
            
            for i in range(1,6):

                P = np.loadtxt(f'./data/FoldedPertubations/P_{q}_{i}e-07_{seed_value}.csv',
                                delimiter=',',dtype='complex_')

                PW = np.einsum('ab,bc->ac',P,W).reshape(q**2,q**2,q**2,q**2)

                df = pd.DataFrame()

                max_dev = np.inf

                for T in range(2*tspan-1):

                    t = float(T)/2

                    data = np.array([])
                    inds = np.array([])

                    for x in range(-T,T+1):
                        x = float(x)/2
                        inds = np.append(inds,x)
                        data = np.append(data,[path_integral(x,t,PW,k=k)])

                    s = pd.Series(data,inds,name=t)

                    df = pd.concat([df, s.to_frame().T])

                    if np.abs(sum(data)) < max_dev: max_dev = sum(data)

                    print('Light cone completely computed for t = ', t, '\n')

                err[i] = max_dev

                print(f'\nFor a pertubation e:{i}e-07 the greatest deviation in charge conservation is: ',
                      max_dev, '\n')
                
                print(df,'\n')

                try:
                    df.to_csv(path+f'heatmap_{q}_{i}e-07_{seed_value}.csv')
                except:
                    os.mkdir(path)
                    df.to_csv(path+f'heatmap_{q}_{i}e-07_{seed_value}.csv')

            err.to_csv(path+'conserved_charge_error.csv')
    
def generate_conefront(q:int,
                        tspan:int,
                        depth:int,
                        k:int = 0,
                        truncate:bool = False):
    
    rstr = r'DU_' + str(q) + r'_([0-9]*).csv'
    rx = re.compile(rstr)

    if truncate:
        path = f"data/truncated_conefront_k{k}/"
    else:
        path = "data/complete_conefront/"

    for _, _, files in os.walk("data/FoldedTensors"):
        for file in files[:]:

            res = rx.match(file)
            seed_value = res.group(1)

            err = pd.Series()
    
            W = np.loadtxt(f'./data/FoldedTensors/DU_{q}_{seed_value}.csv',
                            delimiter=',',dtype='complex_')
            
            for i in range(1,6):

                P = np.loadtxt(f'./data/FoldedPertubations/P_{q}_{i}e-07_{seed_value}.csv',
                                delimiter=',',dtype='complex_')

                PW = np.einsum('ab,bc->ac',P,W).reshape(q**2,q**2,q**2,q**2)

                df = pd.DataFrame()

                max_dev = np.inf

                for T in range(2*tspan-1):

                    t = float(T)/2

                    data = np.array([])
                    inds = np.array([])

                    for x in range(T-depth,T+1):
                        x = float(x)/2
                        inds = np.append(inds,x)
                        data = np.append(data,[path_integral(x,t,PW,k=k)])

                    s = pd.Series(data,inds,name=t)

                    df = pd.concat([df, s.to_frame().T])

                    if np.abs(sum(data)) < max_dev: max_dev = sum(data)

                    print('Light cone front completely computed for t = ', t, '\n')

                err[i] = max_dev

                print(f'\nFor a pertubation e:{i}e-07 the greatest deviation in charge conservation is: ',
                      max_dev, '\n')
                
                print(df,'\n')

                try:
                    df.to_csv(path+f'heatmap_{q}_{i}e-07_{seed_value}.csv')
                except:
                    os.mkdir(path)
                    df.to_csv(path+f'heatmap_{q}_{i}e-07_{seed_value}.csv')

            err.to_csv(path+'conserved_charge_error.csv')

def path_integral(x:float,
                  t:float,
                  W:np.ndarray,
                  k:int = 0):

    def transfer_matrix(W: np.ndarray, a: np.ndarray,
                        x: int, b: np.ndarray = [],
                        horizontal: bool = True,
                        terminate: bool = False,
                        X: float = 0):

        """
            A transfer matrix can either be horizontal or vertical row of contracted
            folded tensors. 
            For the case of a horizontal transfer matrix the row can only be terminated
            either by a defect or by the operator b

            a can be like [0,1,0,0]
        """

        if horizontal:
            direct = W[0,:,:,0]
            defect = W[:,0,:,0]
        else:
            direct = W[:,0,0,:]
            defect = W[0,:,0,:]

        if x > 1:
            for _ in range(x-2):
                a = np.einsum('ba,a->b',direct,a)

        if not terminate:
            a = np.einsum('ba,a->b',direct,a)
            return np.einsum('ba,a->b',defect,a)
        else:
            if int(2*(t+X))%2 == 0:
                a = np.einsum('ba,a->b',direct,a)
            else:
                a = np.einsum('ba,a->b',defect,a)
            return np.einsum('a,a->',b,a)
        
    def skeleton(x_h:np.ndarray, x_v:np.ndarray, W:np.ndarray,
                a:np.ndarray, b:np.ndarray, X:float = 0):
        '''
            Computes a contribution to the path integral of the
            input skeleton diagram
        '''

        for i in range(len(x_v)):
            a = transfer_matrix(W,a,x_h[i])
            a = transfer_matrix(W,a,x_v[i],horizontal=False)

        return transfer_matrix(W,a,x_h[-1],b=b,terminate=True,X=X) if len(x_h) > len(x_v) else np.einsum('a,a->',a,b)

    a, b = np.array([0,1,0,0],dtype="complex_"), np.array([0,1,0,0],dtype="complex_")

    if x == 0 and t == 0.:
        return np.abs(np.einsum('a,a->',a,b))

    vertical_data = {}
    horizontal_data = {}

    x_h, x_v = math.ceil(t+x), math.floor(t+1-x)

    if k > x_h or k > x_v:
        k = min(x_h,x_v)

    list_generator(x_v-1,vertical_data,k=k)
    list_generator(x_h,horizontal_data,k=k)

    n = 1
    sum = 0

    while n <= k:

        try:
            l1, l2 = horizontal_data[n], vertical_data[n]
            for h in l1:
                for v in l2:
                    sum += skeleton(h,v,W,a,b,X=x)
        except:pass
            
        try:
            l1, l2 = horizontal_data[n+1], vertical_data[n]
            for h in l1:
                for v in l2:
                    sum += skeleton(h,v,W,a,b,X=x)
        except:pass
                        
        try:
            l1, v = horizontal_data[1], vertical_data[0]
            for h in l1:
                sum += skeleton(h,[],W,a,b,X=x)
        except:pass

        n += 1

    return np.abs(sum)

def list_generator(x:int,data:dict,k:int=np.inf,
                    lists:np.ndarray=[]):
    '''
        Generates a complete set of possible lists which can
        combine to form a complete set 

        For now just taking base case of the operator a being 
        intially being localised to an integer position 
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

def metropolis_hastings(input:np.ndarray,x:int):

    if input[-1] != x or input[0] != 0:
        raise Exception('Input array not of correct format')
    
    for i in range(1,len(input)-1):
         if random.randrange(sys.maxsize) % 2 and input[i-1]-input[i+1] == 0:
            input[i] = 2*input[i+1] - input[i]

    return input
    
if __name__ == '__main__':
    main()