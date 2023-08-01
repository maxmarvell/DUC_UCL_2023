import numpy as np
from time import time

'''
    Computing the correlation function only where opertor a is initially localised to x 
    , where x is an integer, and operator b is localised to some point y at time 2t, 
    where y is an integer.

    TODO: 1. maybe implement random walk here

    LATER: 1. Possible SVD for larger transfer matricies??
'''

def main():
    data = []
    start = time()
    list_generator(20,data)
    end = time()
    print(len(data))
    print(f"time to run: {end-start}s")

def transfer_matrix(W: np.ndarray, a: np.ndarray, x: int,
                    horizontal: bool = True, terminate: bool = False):

    """
        A transfer matrix can either be horizontal or vertical row of contracted
        folded tensors. 
        For the case of a horizontal transfer matrix the row can only be terminated
        either by a defect or by the operator b

        a can be like [0,1,0,0]
    """

    if horizontal:
        direct = W[0,:,:,0]
        defect = W[:,:,0,0]
    else:
        direct = W[:,0,0,:]
        defect = W[0,0,:,:]

    p = np.einsum('ba,a->b',direct,a)

    for _ in range(x-1):
        p = np.einsum('ba,a->b',direct,p)

    if not terminate:
        return np.einsum('ba,a->b',defect,p)
    else:
        return np.einsum('a,a->',a,p)
    
def skeleton(x_h:np.ndarray, x_v:np.ndarray, W:np.ndarray,
             a:np.ndarray, b:np.ndarray):
    '''
        Computes a contribution to the path integral of the
        input skeleton diagram
    '''

    c = len(x_h)

    for i in range(c-1):
        a = transfer_matrix(W,a,x_h[i])
        a = transfer_matrix(W,a,x_v[i],horizontal=False)

    return transfer_matrix(W,a,x_h[-1],terminate=True)

def list_generator(x:int,data:np.ndarray=[],
                   k:int=np.inf,lists:np.ndarray=[]):
    '''
        Generates a complete set of possible lists which can
        combine to form a complete set 

        For now just taking base case of the operator a being 
        intially being localised to an integer position 
    '''

    if x == 0:
        data.append(lists)
        return
    elif len(lists) >= k:
        return 

    for i in range(1,x+1):
        sublist = lists.copy()
        sublist.append(i)
        list_generator(x-i,data,k,sublist)

def metropolis_hastings():
    pass
    
if __name__ == '__main__':
    main()