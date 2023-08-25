import matplotlib.pyplot as plt
from path_integral import list_generator
from P_generator import Exponential_Map
from path_integral import path_integral as PI
from time import time
import numpy as np
import pandas as pd
import math

def main():
    e = 1e-7
    q = 2
    W_path = 'data/FoldedTensors/DU_2_33296628.csv'
    P_path = 'data/FoldedPertubations/P_2_866021931.csv'
    W = np.loadtxt(W_path,delimiter=',',dtype='complex_')
    G = np.loadtxt(P_path,delimiter=',',dtype='complex_')
    P = Exponential_Map(e,G)
    PW = np.einsum('ab,bc->ac',P,W).reshape([q**2,q**2,q**2,q**2])

    tspan = 12

    tree = Tree()
    df = pd.DataFrame()

    start = time()

    err = pd.Series()

    for T in range(2*tspan+1):
        t = float(T)/2

        data = np.array([])
        inds = np.array([])

        for x in range(-T+1,T+1):
            x = float(x)/2
            inds = np.append(inds,x)
            data = np.append(data,[path_integral(x,t,PW,tree)])

        print(f'computed for T = {t}s')

        s = pd.Series(data,inds,name=t)        
        df = pd.concat([df, s.to_frame().T])
        err[t] = np.abs(sum(data))

    df = df.reindex(sorted(df.columns,key=lambda num: float(num)), axis=1)
    df = df.fillna(0)

    # plt.imshow(df.iloc[::-1],cmap='hot',interpolation='nearest')
    # plt.show()
    # plt.close()

    # end = time()

    # print(f'Time for tree path integral: {end-start}')  

    # df = df.to_numpy(dtype='complex_')

    # cross_section = list(np.abs(df[5,:])**2)
    # space = list((np.array(range(len(cross_section)))  + 1 - (len(cross_section)/2))/2)
    # space_even = space[::2]
    # space_odd = space[1::2]
    # cross_section_even = cross_section[::2]
    # cross_section_odd = cross_section[1::2]

    # plt.scatter(space_odd, cross_section_odd, marker="d", color='k', s=5)

    print(err)

    # plt.show()



    # tree.return_root()

    # printTree(tree.current)      

def path_integral(x:float,
                  t:float,
                  W:np.ndarray,
                  tree,
                  k:int = None):
    
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
            # print('    TERMINATING WITH DIRECT')
            # print(f'      Computing transfer matrix len {l}, Horizontal? {horizontal}')
            for _ in range(l):
                a = np.einsum('ab,b->a',direct,a)
            return np.einsum('a,a->',b,a)

        else:
            # print('      TERMINATING WITH DEFECT')
            # print(f'      Computing transfer matrix len {l}, Horizontal? {horizontal}')
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
            return np.einsum('a,a->',b,a)
        
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
                # print(f'    Followed horizontal transfer matrix of length {h[i]}')
                search_tree(v[i],horizontal=False)
                # print(f'    Followed vertical transfer matrix of length {v[i]}')
            search_tree(h[-1])
            # print(f'    Followed horizontal transfer matrix of length {h[-1]}')

            a = tree.current.data

        elif len(h) > len(v):
            for i in range(len(v)):
                search_tree(h[i])
                # print(f'    Followed horizontal transfer matrix of length {h[i]}')
                search_tree(v[i],horizontal=False)
                # print(f'    Followed vertical transfer matrix of length {v[i]}')

            a = tree.current.data

        return transfer_matrix(a,h[-1]) if len(h) > len(v) else transfer_matrix(a,v[-1],horizontal=False)

    a, b = np.array([0,1,0,0],dtype='complex_'), np.array([0,1,0,0],dtype='complex_')

    if x == 0 and t == 0.:
        return np.einsum('a,a->',a,b)

    vertical_data = {}
    horizontal_data = {}

    x_h, x_v = math.ceil(t+x), math.floor(t+1-x)

    # print(f'\nTIME {t}, POSITION {x} we have x_h:{x_h}, x_v:{x_v}')

    if not k:
        k = min(x_h,x_v)

    list_generator(x_v-1,vertical_data,k=k)
    list_generator(x_h,horizontal_data,k=k+1)

    n = 1
    sum = 0

    while n <= k:

        # print(f'\nComputing skeleton diagram for {n} turns!')

        try:
            l1, l2 = horizontal_data[n], vertical_data[n]
            for h in l1:
                for v in l2:
                    # print(f'\n   Diagram h:{h} and v:{v}')
                    sum += skeleton(h,v,a)
        except:pass
            
        try:
            l1, l2 = horizontal_data[n+1], vertical_data[n]
            for h in l1:
                for v in l2:
                    # print(f'\n   Diagram h:{h} and v:{v}')
                    sum += skeleton(h,v,a)
        except:pass
                        
        try:
            l1, v = horizontal_data[n], vertical_data[0]
            for h in l1:
                # print(f'\n   Diagram h:{h} and v:[]')
                sum += skeleton(h,[],a)
        except:pass

        n += 1

    return sum

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