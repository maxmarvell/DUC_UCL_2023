from path_integral import list_generator
from path_integral import path_integral as PI
from QUIMB_exact import exact_contraction
from P_generator import Exponential_Map
from time import time
import quimb.tensor as qtn
import numpy as np 
import math

def main():
    W = np.loadtxt(f'data/FoldedTensors/DU_2_33296628.csv',delimiter=',',dtype='complex_')
    P = np.loadtxt(f'data/FoldedPertubations/P_2_866021931.csv',delimiter=',',dtype='complex_')
    e = 1e-7
    q = 2
    d = 3

    G = Exponential_Map(e,P)
    PW = np.einsum('ab,bc->ac',G,W).reshape(q**2,q**2,q**2,q**2)

    tiles = get_tiles(PW,d)

    I, Z = np.zeros(q**2,dtype='complex_'), np.zeros(q**2,dtype='complex_')
    I[0], Z[1] = 1, 1

    a = np.einsum('a,b,c->abc',I,I,Z).reshape(-1)
    b = np.einsum('a,b,c->abc',Z,I,I).reshape(-1)

    start = time()
    val = path_integral(0.5,8.5,d,tiles,a,b,K=10)
    end = time()
    print(f'\nTruncated case: {val} computed in {end-start}s')

    # start = time()
    # val = path_integral(0.5,8.5,d,tiles,a,b)
    # end = time()

    # print(f'\nComplete case: {val} computed in {end-start}s')

    # print((PI(0,5.5,PW)))

    # print((exact_contraction(0,5.5,q,PW)))

def path_integral(x:float,
                  t:float,
                  d:int,
                  tiles:dict,
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

        if horizontal:
            direct = tiles['h_direct']
            defect = tiles['h_defect']
        else:
            direct = tiles['v_direct']
            defect = tiles['v_defect']

        if not terminate:
            print(f'      Computing transfer matrix len {l}, Horizontal? {horizontal}')
            for _ in range(l-1):
                a = np.einsum('ab,b->a',direct,a)
            return np.einsum('ab,b->a',defect,a)

        elif terminate and (int(2*(t+x))%2 == 0):
            print('    TERMINATING WITH DIRECT')
            print(f'      Computing transfer matrix len {l}, Horizontal? {horizontal}')
            for _ in range(l):
                a = np.einsum('ab,b->a',direct,a)
            return np.einsum('a,a->',a,b)

        else:
            print('      TERMINATING WITH DEFECT')
            print(f'      Computing transfer matrix len {l}, Horizontal? {horizontal}')
            for _ in range(l-1):
                a = np.einsum('ab,b->a',direct,a)
            a = np.einsum('ab,b->a',defect,a)
            return np.einsum('a,a->',a,b)
        
    def truncated_transfer_matrix(a:np.ndarray,
                                  l:int,
                                  horizontal:bool = True,
                                  terminate:bool = False):
        
        '''
            A transfer matrix can either be horizontal or vertical row of contracted
            folded tensors. 
            For the case of a horizontal transfer matrix the row can only be terminated
            either by a defect or by the operator b
        '''

        def contract(a,u,s,v):
            a = np.einsum('ab,b->a',v,a)
            a = np.multiply(s,a)
            return np.einsum('ab,b->a',u,a)

        if horizontal:
            direct = tiles['h_direct']
            defect = tiles['h_defect']
        else:
            direct = tiles['v_direct']
            defect = tiles['v_defect']

        if K:
            u1, s1, v1 = np.linalg.svd(direct)
            u1, s1, v1 = u1[:,:K], s1[:K], v1[:K,:]

            u2, s2, v2 = np.linalg.svd(defect)
            u2, s2, v2 = u2[:,:K], s2[:K], v2[:K,:]


        if not terminate:
            print(f'      Computing transfer matrix len {l}, Horizontal? {horizontal}')
            for _ in range(l-1):
                a = contract(a,u1,s1,v1)
            return contract(a,u2,s2,v2)

        elif terminate and (int(2*(t+x))%2 == 0):
            print('    TERMINATING WITH DIRECT')
            print(f'      Computing transfer matrix len {l}, Horizontal? {horizontal}')
            for _ in range(l):
                a = contract(a,u1,s1,v1)
            return np.einsum('a,a->',a,b)

        else:
            print('      TERMINATING WITH DEFECT')
            print(f'      Computing transfer matrix len {l}, Horizontal? {horizontal}')
            for _ in range(l-1):
                a = contract(a,u1,s1,v1)
            a = contract(a,u2,s2,v2)
            return np.einsum('a,a->',a,b)
        
    def skeleton(h:np.ndarray,
                 v:np.ndarray,
                 a:np.ndarray):
        '''
            Computes a contribution to the path integral of the
            input skeleton diagram
        '''

        if not K:
            fun = transfer_matrix
        else:
            fun = truncated_transfer_matrix

        for i in range(len(v)):

            a = fun(a,h[i])

            print(f'    Computed horizontal transfer matrix of length {h[i]}')

            a = fun(a,v[i],horizontal=False)

            print(f'    Computed vertical transfer matrix of length {v[i]}')

        return fun(a,h[-1],terminate=True) if len(h) > len(v) else np.einsum('a,a->',a,b)
    
    vertical_data = {}
    horizontal_data = {}

    x_h, x_v = math.ceil(t+x), math.floor(t+1-x)

    x_h /= d
    x_v /= d

    k = min(x_h,x_v)

    list_generator(int(x_v-1),vertical_data,k=k)
    list_generator(int(x_h),horizontal_data,k=k+1)

    n = 1
    sum = 0

    while n < k:

        print(f'\nComputing skeleton diagram for {n} turns!')

        try:
            l1, l2 = horizontal_data[n], vertical_data[n]
            for h in l1:
                for v in l2:
                    print(f'\n   Diagram h:{h} and v:{v}')
                    sum += skeleton(h,v,a)
                    print(f'             Running sum {sum}')
        except:pass
            
        try:
            l1, l2 = horizontal_data[n+1], vertical_data[n]
            for h in l1:
                for v in l2:
                    print(f'\n   Diagram h:{h} and v:{v}')
                    sum += skeleton(h,v,a)
                    print(f'             Running sum {sum}')
        except:pass
                        
        try:
            l1, v = horizontal_data[n], vertical_data[0]
            for h in l1:
                print(f'\n   Diagram h:{h} and v:{v}')
                sum += skeleton(h,[])
                print(f'             Running sum {sum}')
        except:pass

        n += 1

    return sum

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
    print(val)

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
    print(val)

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
    print(val)

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
    print(val)

    v_defect = val.data
    v_defect = v_defect.reshape(-1,*v_defect.shape[-d:])
    v_defect = v_defect.reshape(*v_defect.shape[:1],-1)

    tiles['v_defect'] = v_defect

    return tiles

if __name__ == '__main__':
    main()