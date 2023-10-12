from path_integral import list_generator
import numpy as np
import pandas as pd
import math

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

    def return_root(self):
        self.current = self.root     

def PATH(x:float,
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

        if len(v) == len(h):
            for i in range(len(v)-1):
                search_tree(h[i])
                search_tree(v[i],horizontal=False)
            search_tree(h[-1])

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

    n = 0
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

        n += 1
    return sum


def randPATH(x:float,
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

            return a[1]

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

        return a[1]

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

    a = np.zeros(q**2,dtype='complex_')
    a[1] = 1

    if x == 0 and t == 0.:
        a[1]
    
    vertical_data = {}
    horizontal_data = {}

    x_h, x_v = math.ceil(t+x), math.floor(t+1-x)
    
    k = min(x_h,x_v)

    list_generator(x_v-1,vertical_data,k=k)
    list_generator(x_h,horizontal_data,k=k+1)

    n = 0
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

        n += 1

    return sum