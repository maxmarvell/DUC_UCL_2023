import numpy as np
from time import time
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

    q = 2
    e = 0.1

    rx = re.compile(r'DU_2_([0-9]*).csv')

    for _, _, files in os.walk("data/FoldedTensors"):
        for file in files[:1]:
            res = rx.match(file)
            seed_value = res.group(1)
            random.seed(seed_value)
    
            W = np.loadtxt(f'./data/FoldedTensors/DU_{q}_{seed_value}.csv',
                            delimiter=',',dtype='complex_')
            P = np.loadtxt(f'./data/FoldedPertubations/P_{q}_{e}_{seed_value}.csv',
                            delimiter=',',dtype='complex_')

            pW = np.einsum('ab,bc->ac',P,W)

            print(path_integral(5,9,pW.reshape(q**2,q**2,q**2,q**2)))


def path_integral(x:float,t:int,W:np.ndarray):

    def transfer_matrix(W: np.ndarray, a: np.ndarray,
                        x: int, b: np.ndarray = [],
                        horizontal: bool = True,
                        terminate: bool = False):

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

        p = np.einsum('ba,a->b',direct,a)

        for _ in range(x-1):
            p = np.einsum('ba,a->b',direct,p)

        if not terminate and not horizontal:
            return np.einsum('ba,a->b',defect,p)
        elif not terminate and horizontal:
            return p
        else:
            return np.einsum('a,a->',b,p)
        
    def skeleton(x_h:np.ndarray, x_v:np.ndarray, W:np.ndarray,
                a:np.ndarray, b:np.ndarray):
        '''
            Computes a contribution to the path integral of the
            input skeleton diagram
        '''

        for i in range(len(x_v)):
            a = transfer_matrix(W,a,x_h[i])
            a = transfer_matrix(W,a,x_v[i],horizontal=False)

        return np.abs(transfer_matrix(W,a,x_h[-1],b=b,terminate=True)) if len(x_h) > len(x_v) else np.abs(np.einsum('a,a->',a,b))

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

    vertical_data = {}
    horizontal_data = {}

    x_h, x_v = t + math.ceil(x), t + 1 - math.ceil(x)

    k = min(x_h,x_v)

    list_generator(x_v,vertical_data,k=k)
    list_generator(x_h,horizontal_data,k=k)

    n = 1
    sum = 0
    a, b = np.array([0,1,0,0],dtype="complex_"), np.array([0,1,0,0],dtype="complex_")

    while n <= k:

        l1, l2 = horizontal_data[n], vertical_data[n]

        for h in l1:
            for v in l2:
                sum += skeleton(h,v,W,a,b)

        if n - 1:
            l1, l2 = horizontal_data[n], vertical_data[n-1]
            for h in l1:
                for v in l2:
                    sum += skeleton(h,v,W,a,b)
        
        n += 1

    a = transfer_matrix(W,a,4)

    return sum

def metropolis_hastings(input:np.ndarray,x:int):

    if input[-1] != x or input[0] != 0:
        raise Exception('Input array not of correct format')
    
    for i in range(1,len(input)-1):
         if random.randrange(sys.maxsize) % 2 and input[i-1]-input[i+1] == 0:
            input[i] = 2*input[i+1] - input[i]

    return input
    
if __name__ == '__main__':
    main()