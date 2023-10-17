from utils import *
import numpy as np
import pandas as pd
import math

'''
    Code developed to compute two-point correlation function in sub-complete contraction 
    time.
'''

def PATH(x:float,
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
                  circuit:pd.DataFrame,
                  PW:dict,
                  k:int = 0):
    
    '''
        Computes path integral corresponding to input circuit arrangement.
        Neglects contribution from dressed/thicker paths and looping paths
        as in Ref:________

        args:
            x: half-int float (.5)
                The relative position of the second operator
            t: half-int float (.5)
                The time take to evolve the local operator
            circuit: pd.DataFrame
                A DataFrame encoding the gate locations, where each gate is stored
                under a seed name
            PW: dict
                Dictionary mapping seeds to discrete set of peturbed DU gates
        
        kwargs:
            k: int
                Maximum number of partitions considered when using list generator
                see 'path_integral.py' for details

        returns:
            sum: complex
                Path integral approximation to two-point correlation fucntion
    '''

    def transfer_matrix(a:np.ndarray,
                        l:int,
                        X:float,
                        T:float,
                        horizontal:bool = True,
                        terminate:bool = False):

        '''
            Program that multiplies initial operator by succesive gates (transfer
            matrix)

            args:
                a: np.ndarray
                    Input local operator to transfer matrix
                l: int
                    Length of the transfer matrix
                X: float
                    Initial position of the local operator prior
                T: float
                    Initial time of the local operator prior

            kwargs:
                horizontal: bool
                    True if the transfer matrix is horizontal in the 45deg rotated picture
                terminate: bool
                    True where path is contracted with second point operator, effectively 
                    terminating and returning a singular value

            return:
                a: np.ndarray
                    Where there are still open output legs a is returned to skeleton fun
                or
                sum: complex
                    Where terminated returns contribution of path to path integral
                
        '''

        if not terminate:

            for _ in range(l-1):
                W = PW[circuit.loc[T,X]]
                if horizontal:
                    X = X + 0.5
                    T = T + 0.5
                    direct = W[0,:,:,0]
                else:
                    X = X - 0.5
                    T = T + 0.5
                    direct = W[:,0,0,:]
                a = np.einsum('ab,b->a',direct,a)

            W = PW[circuit.loc[T,X]]

            if horizontal:
                T = T + 0.5
                defect = W[:,0,:,0]
            else:
                T = T + 0.5
                defect = W[0,:,0,:]

            return a[1]

        elif ((int(2*(t+x))%2 == 0) and horizontal) or ((int(2*(t+x))%2 != 0) and not horizontal):

            for _ in range(l):
                W = PW[circuit.loc[T,X]]
                if horizontal:
                    X = X + 0.5
                    T = T + 0.5
                    direct = W[0,:,:,0]
                else:
                    X = X - 0.5
                    T = T + 0.5
                    direct = W[:,0,0,:]

                a = np.einsum('ab,b->a',direct,a)
            return a[1]

        for _ in range(l-1):
            W = PW[circuit.loc[T,X]]
            if horizontal:
                X = X + 0.5
                T = T + 0.5
                direct = W[0,:,:,0]
            else:
                X = X - 0.5
                T = T + 0.5
                direct = W[:,0,0,:]

            a = np.einsum('ab,b->a',direct,a)

        W = PW[circuit.loc[T,X]]

        if horizontal:
            T = T + 0.5
            defect = W[:,0,:,0]
        else:
            T = T + 0.5
            defect = W[0,:,0,:]

        a = np.einsum('ab,b->a',defect,a)
        return a[1]
        
    def skeleton(h:np.ndarray,
                 v:np.ndarray,
                 a:np.ndarray):
        '''
            Computes a contribution to the path integral of the
            input skeleton diagram
        '''

        X,T = 0,0
        
        if len(v) == len(h):
            for i in range(len(v)-1):

                a = transfer_matrix(a,h[i],X,T)
                X += (h[i] - 1) / 2
                T += h[i]/2

                a = transfer_matrix(a,v[i],X,T,horizontal=True)
                X -= (h[i] - 1) / 2
                T += v[i]/2

            a = transfer_matrix(a,h[-1],X,T)
            X += (h[-1] - 1) / 2
            T += h[-1]/2
            
            return transfer_matrix(a,v[-1],X,T,terminate=True,horizontal=False)
        
        for i in range(len(v)):

            a = transfer_matrix(a,h[i],X,T)
            X += (h[i] - 1) / 2
            T += h[i]/2

            a = transfer_matrix(a,v[i],X,T,horizontal=False)
            X -= (v[i] - 1) / 2
            T += v[i]/2

        return transfer_matrix(a,h[-1],X,T,terminate=True)

    a = np.array([0,1,0,0],dtype='complex_')

    if x == 0 and t == 0:
        return a[1]

    vertical_data = {}
    horizontal_data = {}

    x_h, x_v = math.ceil(t+x), math.floor(t+1-x)

    if k > x_h or k > x_v or not k:
        k = min(x_h,x_v)

    list_generator(x_v-1,vertical_data,k=k)
    list_generator(x_h,horizontal_data,k=k)

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