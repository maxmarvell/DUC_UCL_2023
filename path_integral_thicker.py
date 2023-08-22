from path_integral import list_generator
from P_generator import Exponential_Map
import quimb.tensor as qtn
import numpy as np 
import math

def path_integral(x:float,
                  t:float,
                  d:int,
                  W:np.ndarray):
    
    def transfer_matrix(a:np.ndarray,
                        x:int,
                        horizontal:bool = True,
                        terminate:bool = False):

        '''
            A transfer matrix can either be horizontal or vertical row of contracted
            folded tensors. 
            For the case of a horizontal transfer matrix the row can only be terminated
            either by a defect or by the operator b

            a can be like [0,1,0,0]
        '''

        # if horizontal:
        #     direct = W[0,:,:,0]
        #     defect = W[:,0,:,0]
        # else:
        #     direct = W[:,0,0,:]
        #     defect = W[0,:,0,:]

        if horizontal:
            pass

        if not terminate:
            for _ in range(x-d):
                a = np.einsum('ab,b->a',direct,a)
            return np.einsum('ab,b->a',defect,a)

        elif terminate and (int(2*(t+X))%2 == 0):
            for _ in range(x):
                a = np.einsum('ab,b->a',direct,a)
            return np.einsum('a,a->',b,a)

        else:
            for _ in range(x-d):
                a = np.einsum('ab,b->a',direct,a)
            a = np.einsum('ab,b->a',defect,a)
            return np.einsum('a,a->',b,a)
        
    def skeleton(h:np.ndarray,
                 v:np.ndarray):
        '''
            Computes a contribution to the path integral of the
            input skeleton diagram
        '''

        for i in range(len(x_v)):
            a = transfer_matrix(a,h[i])
            a = transfer_matrix(a,v[i],horizontal=False)

        return transfer_matrix(W,a,h[-1],terminate=True) if len(h) > len(v) else np.einsum('a,a->',a,b)
    
    
    vertical_data = {}
    horizontal_data = {}

    x_h, x_v = math.ceil(t+x), math.floor(t+1-x)

    k = min(x_h//d,x_v//d)

    x_h = x_h - k*(d-1)

    if int(2*(t+x))%2 == 0:
        x_v = x_v - (k)*(d-1) - d
    else:
        x_v = x_v - (k)*(d-1) - 1

    n = 1
    sum = 0

    while n <= k:

        try:
            l1, l2 = horizontal_data[n], vertical_data[n]
            for h in l1:
                for v in l2:
                    sum += skeleton(h,v)
        except:pass
            
        try:
            l1, l2 = horizontal_data[n+1], vertical_data[n]
            for h in l1:
                for v in l2:
                    sum += skeleton(h,v)
        except:pass
                        
        try:
            l1, v = horizontal_data[n], vertical_data[0]
            for h in l1:
                sum += skeleton(h,[])
        except:pass

        n += 1

    return sum

def get_tiles(W:np.ndarray,
              d:int):

    ### HORIZONTAL ###

    tensors = np.array([])
    
    tensors = np.append(tensors,[qtn.Tensor(W[:,:,:,0],inds=('k1,2','k2,1','k0,1'))])
    tensors = np.append(tensors,[qtn.Tensor(W,inds=(f'k1,{2*j+2}',f'k2,{2*j+1}',
                                                        f'k0,{2*j+1}',f'k1,{2*j}')) for j in range(1,d-1)])
    tensors = np.append(tensors,[qtn.Tensor(W[0,:,:,:],inds=(f'k2,{2*(d-1)+1}',f'k0,{2*(d-1)+1}',f'k1,{2*(d-1)}'))])

    TN = qtn.TensorNetwork(tensors)
    # TN.draw()

    val = TN.contract()

    reshape = tuple()

    for i in range(d):
        reshape = reshape + (f'k0,{2*i+1}',)
    for i in range(d):
        reshape = reshape + (f'k2,{2*i+1}',)

    val = val.transpose(*reshape,inplace=True)

    print(val)

    h_direct = val.data

    tensors = np.array([])

    tensors = np.append(tensors,[qtn.Tensor(W[:,:,:,0],inds=(f'k{2*i+1},2',f'k{2*i+2},1',f'k{2*i},1')) for i in range(d-1)])
    tensors = np.append(tensors,[qtn.Tensor(W[:,0,:,0],inds=(f'k{2*(d-1)+1},2',f'k{2*(d-1)},1'))])

    for j in range(1,d-1):

        tensors = np.append(tensors,[qtn.Tensor(W,inds=(f'k{2*i+1},{2*j+2}',f'k{2*i+2},{2*j+1}',
                                                        f'k{2*i},{2*j+1}',f'k{2*i+1},{2*j}')) for i in range(d-1)])
        
        tensors = np.append(tensors,[qtn.Tensor(W[:,0,:,:],inds=(f'k{2*(d-1)+1},{2*j+2}',f'k{2*(d-1)},{2*j+1}',f'k{2*(d-1)+1},{2*j}'))])

    tensors = np.append(tensors,[qtn.Tensor(W,inds=(f'k{2*i+1},{2*(d-1)+2}',f'k{2*i+2},{2*(d-1)+1}',
                                                        f'k{2*i},{2*(d-1)+1}',f'k{2*i+1},{2*(d-1)}')) for i in range(d-1)])
        
    tensors = np.append(tensors,[qtn.Tensor(W[:,0,:,:],inds=(f'k{2*(d-1)+1},{2*(d-1)+2}',f'k{2*(d-1)},{2*(d-1)+1}',f'k{2*(d-1)+1},{2*(d-1)}'))])

    

    TN = qtn.TensorNetwork(tensors)
    # TN.draw()

    val = TN.contract()

    reshape = tuple()

    for i in range(d):
        reshape = reshape + (f'k0,{2*i+1}',)
    for i in range(d):
        reshape = reshape + (f'k{2*i+1},{2*d}',)

    print(val)

    h_defect = val.data

    ### VERTICAL ###

    tensors = np.array([])
    
    tensors = np.append(tensors,[qtn.Tensor(W[:,:,0,:],inds=('k1,2','k2,1','k1,0'))])
    tensors = np.append(tensors,[qtn.Tensor(W,inds=(f'k{2*i+1},2',f'k{2*i+2},1',f'k{2*i},1',f'k{2*i+1},0')) for i in range(1,d-1)])
    tensors = np.append(tensors,[qtn.Tensor(W[:,0,:,:],inds=(f'k{2*(d-1)+1},2',f'k{2*(d-1)},1',f'k{2*(d-1)+1},0'))])

    TN = qtn.TensorNetwork(tensors)

    val = TN.contract()

    reshape = tuple()

    for i in range(d):
        reshape = reshape + (f'k{2*i+1},0',)
    for i in range(d):
        reshape = reshape + (f'k{2*i+1},2',)

    val = val.transpose(*reshape,inplace=True)

    print(val)

    v_direct = val.data

    tensors = np.array([])

    tensors = np.append(tensors,[qtn.Tensor(W[:,:,0,:],inds=('k1,2','k2,1','k1,0'))])
    tensors = np.append(tensors,[qtn.Tensor(W,inds=(f'k{2*i+1},2',f'k{2*i+2},1',
                                                    f'k{2*i},1',f'k{2*i+1},0')) for i in range(1,d)])

    for j in range(1,d-1):
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

    return h_direct, h_defect, v_direct, v_defect

W = np.loadtxt(f'Sample_Tensor.csv',delimiter=',',dtype='complex_')
P = np.loadtxt(f'Sample_Perturbation.csv',delimiter=',',dtype='complex_')
e = 1e-7
q = 2

G = Exponential_Map(e,P)
PW = np.einsum('ab,bc->ac',G,W).reshape(q**2,q**2,q**2,q**2)


h_dir,h_def,v_dir,v_def = get_tiles(PW,3)

