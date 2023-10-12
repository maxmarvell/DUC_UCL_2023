from utils import *
import numpy as np
import math

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