import matplotlib.pyplot as plt
from path_integral import list_generator
from P_generator import Exponential_Map
from time import time
import numpy as np
import pandas as pd
import math
import os
import re

def main():

    pertubations = [1e-7,3e-7,5e-7]
    q = 2
    tspan = 11
    k = 3

    for e in pertubations:
        generate_data(q,tspan,e)

class Node:
    def __init__(self,data:complex=None):
        self.left = None
        self.right = None
        self.data = data

class Tree:
    def __init__(self,root=Node()):
        self.root = root
        self.current = self.root

    def move_left(self):
        if not self.current.left:
            self.current.left = Node()
        self.current = self.current.left

    def move_right(self):
        if not self.current.right:
            self.current.right = Node()
        self.current = self.current.right
    
    def assign_data(self,data:complex):
        self.current.data = data
        # print(f'                        DATA ASSIGNED')

    def return_root(self):
        self.current = self.root     

def generate_data(q:int,
                  tspan:int,
                  e:float,
                  k:int = np.inf):
    
    if k != np.inf:
        path = 'data/PathIntegralTreeTruncated'
    else:
        path = 'data/PathIntegralTree'

    rstr = r'DU_' + str(q) + r'_([0-9]*).csv'
    rx = re.compile(rstr)

    for _, _, files in os.walk('data/FoldedTensors'):
        for file in files[:1]:

            res = rx.match(file)

            if not res:
                continue

            seed = res.group(1)
    
            W = np.loadtxt(f'data/FoldedTensors/DU_{q}_{seed}.csv',
                            delimiter=',',dtype='complex_')

            G = np.loadtxt(f'data/FoldedPertubations/P_{q}_866021931.csv',
                           delimiter=',',dtype='complex_')

            P = Exponential_Map(e,G)
            PW = np.einsum('ab,bc->ac',P,W).reshape([q**2,q**2,q**2,q**2])

            tree = Tree()
            df = pd.DataFrame()

            start = time()

            err = pd.Series()

            for T in range(2*tspan+1):

                t = float(T)/2

                if t == 0:
                    s = pd.Series(np.array([1],dtype='complex_'),np.array([0]),name=t)        
                    df = pd.concat([df, s.to_frame().T])
                    err[t] = 1
                    print(f'\ncomputed for T = {t}s')
                    continue

                data = np.array([])
                inds = np.array([])

                for x in range(-T+1,T+1):
                    x = float(x)/2
                    inds = np.append(inds,x)
                    data = np.append(data,[path_integral(x,t,q,PW,tree,k=k)])

                print(f'\ncomputed for T = {t}s')

                s = pd.Series(data,inds,name=t)        
                df = pd.concat([df, s.to_frame().T])
                err[t] = np.abs(sum(data))

            df = df.reindex(sorted(df.columns,key=lambda num: float(num)), axis=1)
            df = df.fillna(0)
            df = df.iloc[::-1]
            print('\n',df,'\n')

            end = time()

            print(f'Time for tree path integral: {end-start}\n') 

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

def path_integral(x:float,
                  t:float,
                  q:int,
                  W:np.ndarray,
                  tree:Tree,
                  k:int = np.inf):
    
    def transfer_matrix(a:np.ndarray,
                        l:int,
                        horizontal:bool = True):

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

        if ((int(2*(t+x))%2 == 0) and horizontal) or ((int(2*(t+x))%2 != 0) and not horizontal):
            for _ in range(l):
                a = np.einsum('ab,b->a',direct,a)
            return a[1]

        else:
            for _ in range(l-1):

                if horizontal:
                    tree.move_right()
                else:
                    tree.move_left()
                    
                a = np.einsum('ab,b->a',direct,a)

            if not horizontal:
                tree.move_left()
            else:
                tree.move_right()

            a = np.einsum('ab,b->a',defect,a)
            tree.assign_data(a)
            return a[1]
        
    def search_tree(l:int,
                    horizontal:bool = True,):
        
        if horizontal:
            for _ in range(l):
                tree.move_right()
        else:
            for _ in range(l):
                tree.move_left()

    def skeleton(h:np.ndarray,
                 v:np.ndarray,
                 a:np.ndarray):
        '''
            Computes a contribution to the path integral of the
            input skeleton diagram, only for cases where y is an integer!
        '''

        tree.return_root()

        if v == []:
            return transfer_matrix(a,h[-1])

        if len(v) == len(h):
            for i in range(len(v)-1):
                search_tree(h[i])
                search_tree(v[i],horizontal=False)
            search_tree(h[-1])

            a = tree.current.data

        elif len(h) > len(v):
            for i in range(len(v)):
                search_tree(h[i])
                search_tree(v[i],horizontal=False)

            a = tree.current.data

        return transfer_matrix(a,h[-1]) if len(h) > len(v) else transfer_matrix(a,v[-1],horizontal=False)

    a = np.zeros([q**2],dtype='complex_')
    a[1] = 1

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

def printTree(root:Node, level:int=0):

    if root.left:
        print("  " * level, root.left.data)
    if root.right:
        print("  " * level, root.right.data)

    if root.left:
        printTree(root.left, level + 1)
    if root.right:
        printTree(root.right, level + 1)

if __name__ == '__main__':
    main()