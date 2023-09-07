from path_integral import list_generator
from path_integral import path_integral as PI
from QUIMB_exact import exact_contraction
from P_generator import Exponential_Map
from time import time
import quimb.tensor as qtn
import numpy as np 
import math

def main():
    W = np.loadtxt(f'Sample_Tensor.csv',delimiter=',',dtype='complex_')
    P = np.loadtxt(f'Sample_Perturbation.csv',delimiter=',',dtype='complex_')
    e = 4e-7
    q = 2
    d = 3

    G = Exponential_Map(e,P)
    PW = np.einsum('ab,bc->ac',G,W).reshape(q**2,q**2,q**2,q**2)

    I, Z = np.zeros(q**2,dtype='complex_'), np.zeros(q**2,dtype='complex_')
    I[0], Z[1] = 1, 1

    a = np.einsum('a,b,c->abc',I,I,Z).reshape(-1)
    b = np.einsum('a,b,c->abc',Z,I,I).reshape(-1)

    start = time()
    val = path_integral(2.0,4.5,d,a,b,K=10)
    end = time()
    print(f'\nTruncated case: {np.abs(val)} computed in {end-start}s')

    start = time()
    val = path_integral(2.0,4.5,d,a,b)
    end = time()
    print(f'\nComplete case: {np.abs(val)} computed in {end-start}s')

    start = time()
    val = PI(2.0,4.5,PW)
    end = time()
    print(f'\nPath Integral: {np.abs(val)} computed in {end-start}s')

    start = time()
    val = exact_contraction(2.5,4.5,q,PW)
    end = time()
    print(f'\nQUIMB: {(val)} computed in {end-start}s\n')

def path_integral(x:float,
                  t:float,
                  d:int,
                  a:np.ndarray,
                  b:np.ndarray,
                  K:int = None):
    
    def transfer_matrix(a:np.ndarray,
                        l:int,
                        horizontal:bool = True,
                        terminate:bool = False):

        '''
            A transfer matrix can either be horizontal or vertical row of contracted
            folded tensors. 
            For the case of a horizontal transfer matrix the row can only be terminated
            either by a defect or by the operator b
        '''

        def contract(a,A):
            return np.einsum('ab,b->a',A,a)
        
        def truncated_contract(a,u,s,v):
            a = np.einsum('ab,b->a',v,a)
            a = np.multiply(s,a)
            return np.einsum('ab,b->a',u,a)
        
        if not K:
            fun = contract
        else:
            fun = truncated_contract

        if horizontal:
            direct = tiles['h_direct']
            defect = tiles['h_defect']
        else:
            direct = tiles['v_direct']
            defect = tiles['v_defect']

        if not terminate:
            for _ in range(l-1):
                a = fun(a,*direct)
            return fun(a,*defect)

        elif ((int(2*(t+x))%2 == 0) and horizontal) or ((int(2*(t+x))%2 != 0) and not horizontal):
            for _ in range(l):
                a = fun(a,*direct)
            return np.einsum('a,a->',a,b)

        else:
            for _ in range(l-1):
                a = fun(a,*direct)
            a = fun(*defect)
            return np.einsum('a,a->',a,b)
        
    def skeleton(h:np.ndarray,
                 v:np.ndarray,
                 a:np.ndarray):
        '''
            Computes a contribution to the path integral of the
            input skeleton diagram
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
    
    tiles = tile_SVD(d,K)
    
    vertical_data = {}
    horizontal_data = {}

    x_h, x_v = math.ceil(t+x), math.floor(t+1-x)

    x_h /= d
    x_v /= d

    k = int(min(x_h,x_v))

    list_generator(int(x_v-1),vertical_data,k=k)
    list_generator(int(x_h),horizontal_data,k=k+1)

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
            l1, _ = horizontal_data[n], vertical_data[0]
            for h in l1:
                sum += skeleton(h,[])
        except:pass

        n += 1

    return sum

def tile_SVD(d:int,
             K:int):

    tiles = dict()
    path = 'data/TileZoo'
    types = ['h_direct','h_defect','v_direct','v_defect']

    if not K:
        for i in types:
            tiles[i] = (np.loadtxt(f'{path}/{i}_{d}x{d}.csv',delimiter=',',dtype='complex_'),)

    else:
        for i in types:
            tile = np.loadtxt(f'{path}/{i}_{d}x{d}.csv',delimiter=',',dtype='complex_')
            U, S, V = np.linalg.svd(tile)
            tiles[i] = (U[:,:K],S[:K],V[:K,:],)

    return tiles

def get_tiles(W:np.ndarray,
              d:int):

    '''
        gets the four types of tiles corresponding to horizontal defect,
        horizontal direct, vertical defect, vertical direct for a d dimensional
        case
    '''

    tiles = dict()

    if d == 1:
        tiles['h_defect'] = W[:,0,:,0]
        tiles['h_direct'] = W[0,:,:,0]
        tiles['v_defect'] = W[0,:,0,:]
        tiles['v_direct'] = W[:,0,0,:]

    # HORIZONTAL DIRECT TILE

    tensors, inds1, inds2 = np.array([]), tuple(), tuple()
    
    for i in range(d):

        tensors = np.append(tensors,[qtn.Tensor(W[:,:,:,0],inds=(f'k{2*i+1},2',f'k{2*i+2},1',f'k{2*i},1'))])
        tensors = np.append(tensors,[qtn.Tensor(W,inds=(f'k{2*i+1},{2*j+2}',f'k{2*i+2},{2*j+1}',
                                                            f'k{2*i},{2*j+1}',f'k{2*i+1},{2*j}')) for j in range(1,d-1)])
        tensors = np.append(tensors,[qtn.Tensor(W[0,:,:,:],inds=(f'k{2*i+2},{2*(d-1)+1}',f'k{2*i},{2*(d-1)+1}',f'k{2*i+1},{2*(d-1)}'))])

        inds1 = inds1 + (f'k0,{2*i+1}',)
        inds2 = inds2 + (f'k{2*(d-1)+2},{2*i+1}',)

    TN = qtn.TensorNetwork(tensors)
    val = TN.contract()

    reshape = inds1 + inds2
    val = val.transpose(*reshape,inplace=True)

    h_direct = val.data
    h_direct = h_direct.reshape(-1,*h_direct.shape[-d:])
    h_direct = h_direct.reshape(*h_direct.shape[:1],-1)

    tiles['h_direct'] = h_direct

    # HORIZONTAL DEFECT TILE

    tensors = np.array([])

    tensors = np.append(tensors,[qtn.Tensor(W[:,:,:,0],inds=(f'k{2*i+1},2',f'k{2*i+2},1',f'k{2*i},1')) for i in range(d-1)])
    tensors = np.append(tensors,[qtn.Tensor(W[:,0,:,0],inds=(f'k{2*(d-1)+1},2',f'k{2*(d-1)},1'))])

    for j in range(1,d):

        tensors = np.append(tensors,[qtn.Tensor(W,inds=(f'k{2*i+1},{2*j+2}',f'k{2*i+2},{2*j+1}',
                                                        f'k{2*i},{2*j+1}',f'k{2*i+1},{2*j}')) for i in range(d-1)])
        
        tensors = np.append(tensors,[qtn.Tensor(W[:,0,:,:],inds=(f'k{2*(d-1)+1},{2*j+2}',f'k{2*(d-1)},{2*j+1}',f'k{2*(d-1)+1},{2*j}'))])

    TN = qtn.TensorNetwork(tensors)
    val = TN.contract()

    reshape = tuple()
    for i in range(d):
        reshape = reshape + (f'k0,{2*i+1}',)
    for i in range(d):
        reshape = reshape + (f'k{2*i+1},{2*d}',)

    val = val.transpose(*reshape,inplace=True)

    h_defect = val.data
    h_defect = h_defect.reshape(-1,*h_defect.shape[-d:])
    h_defect = h_defect.reshape(*h_defect.shape[:1],-1)

    tiles['h_defect'] = h_defect

    ### VERTICAL DIRECT TILE

    tensors, inds1, inds2 = np.array([]), tuple(), tuple()
        
    for j in range(d):

        tensors = np.append(tensors,[qtn.Tensor(W[:,:,0,:],inds=(f'k1,{2*j+2}',f'k2,{2*j+1}',f'k1,{2*j}'))])
        tensors = np.append(tensors,[qtn.Tensor(W,inds=(f'k{2*i+1},{2*j+2}',f'k{2*i+2},{2*j+1}',f'k{2*i},{2*j+1}',f'k{2*i+1},{2*j}')) for i in range(1,d-1)])
        tensors = np.append(tensors,[qtn.Tensor(W[:,0,:,:],inds=(f'k{2*(d-1)+1},{2*j+2}',f'k{2*(d-1)},{2*j+1}',f'k{2*(d-1)+1},{2*j}'))])

        inds1 = inds1 + (f'k{2*j+1},0',)
        inds2 = inds2 + (f'k{2*j+1},{2*(d-1)+2}',)

    TN = qtn.TensorNetwork(tensors)
    val = TN.contract()

    reshape = inds1 + inds2
    val = val.transpose(*reshape,inplace=True)

    v_direct = val.data
    v_direct = v_direct.reshape(-1,*v_direct.shape[-d:])
    v_direct = v_direct.reshape(*v_direct.shape[:1],-1)

    tiles['v_direct'] = v_direct

    ### VERTICAL DEFECT TILE

    tensors = np.array([])

    for j in range(d-1):
        tensors = np.append(tensors,[qtn.Tensor(W[:,:,0,:],inds=(f'k1,{2*j+2}',f'k2,{2*j+1}',f'k1,{2*j}'))])
        tensors = np.append(tensors,[qtn.Tensor(W,inds=(f'k{2*i+1},{2*j+2}',f'k{2*i+2},{2*j+1}',
                                                        f'k{2*i},{2*j+1}',f'k{2*i+1},{2*j}')) for i in range(1,d)])

    tensors = np.append(tensors,[qtn.Tensor(W[0,:,0,:],inds=(f'k2,{2*(d-1)+1}',f'k1,{2*(d-1)}'))])
    tensors = np.append(tensors,[qtn.Tensor(W[0,:,:,:],inds=(f'k{2*i+2},{2*(d-1)+1}',f'k{2*i},{2*(d-1)+1}',f'k{2*i+1},{2*(d-1)}')) for i in range(1,d)])

    TN = qtn.TensorNetwork(tensors)

    val = TN.contract()

    reshape = tuple()
    for i in range(d):
        reshape = reshape + (f'k{2*i+1},0',)
    for i in range(d):
        reshape = reshape + (f'k{2*d},{2*i+1}',)

    val = val.transpose(*reshape,inplace=True)

    v_defect = val.data
    v_defect = v_defect.reshape(-1,*v_defect.shape[-d:])
    v_defect = v_defect.reshape(*v_defect.shape[:1],-1)

    tiles['v_defect'] = v_defect

    return tiles

if __name__ == '__main__':
    main()