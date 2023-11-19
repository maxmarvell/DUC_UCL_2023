from scipy.linalg import expm 
import quimb.tensor as qtn
import pandas as pd
import numpy as np
import pkg_resources
import math
import os

def load_W():
    '''
        Returns the sample W tensor as a numpy array
    '''
    stream = pkg_resources.resource_stream(__name__, 
                                           'data/sample_tensor.csv')
    return np.loadtxt(stream,dtype='complex_',delimiter=',')

def load_P():
    '''
        Returns the sample P tensor as a numpy array
    '''
    stream = pkg_resources.resource_stream(__name__,
                                           'data/sample_pertubation.csv')
    return np.loadtxt(stream,dtype='complex_',delimiter=',')

def perturb_gate(W:np.ndarray,
                 P:np.ndarray,
                 e:float):
    
    if not W.shape == P.shape:
        raise Exception('Tensors must be of same size!')
    
    q = int(np.sqrt(W.shape[1]))

    return np.einsum('ab,bc->ac',expm(e*P),W).reshape(q,q,q,q)


def infinite_temperature(tspan:int,
                         q:int,
                         path_to_tileset:str,
                         d:int = 3):
    
    '''
        Computes entire light cone of two-point correlation
        functions, under a dual-unitary framework.

        args:
            q: The local Hilbert space dimension aka 2 for qubits
            tspan: How far to probe correlation functions in time
            e: The strength of the pertubation away from the dual-unitarity
            W: A tensor to build the dual unitary floquet structure
            P: A pertubation tensor to apply to W

        kwargs:
            d:
            

        returns:
            df: A pandas dataframe containing the correlation function at
            every half and full integer time step
            err: A dataframe containing accumulative conserved charge 
            error at each time point
    '''
    
    def path_integral(x:float,
                      t:float):
        
        def transfer_matrix(a:np.ndarray,
                            l:int,
                            horizontal:bool = True,
                            status:str = 'Main'):

            '''
                A transfer matrix can either be horizontal or vertical row of contracted
                folded tensors. 
                For the case of a horizontal transfer matrix the row can only be terminated
                either by a defect or by the operator b

                a can be like [0,1,0,0]
            '''

            if horizontal:
                direct = tile_set[status + ' h_direct']
                defect = tile_set[status + ' h_defect']
            else:
                direct = tile_set[status + ' v_direct']
                defect = tile_set[status + ' v_defect']

            if (int(2*(t+x))%2 == 0) and horizontal:
                for i in range(l):
                    if i == 0 and status == 'Low Trim' and (x_h%d) != 0:
                        a = np.einsum('ab,b->a',tile_set[f'Corner h_direct'],a)
                    elif i != 0 and status == 'Left Trim':
                        a = np.einsum('ab,b->a',tile_set['Main h_direct'],a)
                    else:
                        a = np.einsum('ab,b->a',direct,a)
                return a[1]

            elif ((int(2*(t+x))%2 == 0) and not horizontal):
                for _ in range(l-1):
                    a = np.einsum('ab,b->a',direct,a)
                a = np.einsum('ab,b->a',defect,a)
                search_tree(l,status,horizontal)
                tree.assign_data(a)
                return a[1]

            elif ((int(2*(t+x))%2 != 0) and not horizontal):
                for _ in range(l):
                    a = np.einsum('ab,b->a',direct,a)
                return a[1]


            for i in range(l-1):
                if i == 0 and status == 'Low Trim' and (x_h%d) != 0:
                    a = np.einsum('ab,b->a',tile_set[f'Corner h_direct'],a)
                elif i != 0 and status == 'Left Trim':
                    a = np.einsum('ab,b->a',tile_set['Main h_direct'],a)
                else:
                    a = np.einsum('ab,b->a',direct,a)

            if l == 1 and status == 'Low Trim' and (x_h%d) != 0:
                a = np.einsum('ab,b->a',tile_set[f'Corner h_defect'],a)
                search_tree(l,status)
                tree.assign_data(a)
                return a[1]

            elif l > 1 and status == 'Left Trim':
                a = np.einsum('ab,b->a',tile_set['Main h_defect'],a)
                search_tree(l,status)
                tree.assign_data(a)
                return a[1]
                
            a = np.einsum('ab,b->a',defect,a)
            search_tree(l,status)
            tree.assign_data(a)
            return a[1]
        
        def search_tree(l:int,
                        status:str,
                        horizontal:bool = True):   
            
            if status == 'Low Trim':
                for _ in range(l*d+x_h%d-d*int(bool(x_h%d))):
                    tree.move_right()
                for _ in range(x_v%d-1):
                    tree.move_left()
                return
            elif status == 'Left Trim' and horizontal:
                for _ in range(l*d+x_h%d-d*int(bool(x_h%d))):
                    tree.move_right()
                for _ in range(d-1):
                    tree.move_left()
                return
            elif status == 'Left Trim':
                for _ in range(l*d):
                    tree.move_left()
                for _ in range(x_h%d-1):
                    tree.move_right()
                return
            elif status == 'Main' and horizontal:
                for _ in range(l*d):
                    tree.move_right()
                for _ in range(d-1):
                    tree.move_left()
                return
            
            for _ in range(l*d):
                tree.move_left()
            for _ in range(d):
                tree.move_right()
            

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
                    status = 'Low Trim' if ((x_v%d)!=0 and i==0) else 'Left Trim' if ((x_h%d)!=0 and (x_v%d)==0 and i==0) else 'Main'
                    search_tree(h[i],status)

                    status = 'Left Trim' if ((x_h%d)!=0 and h[0]==1 and i==0) else 'Main'
                    search_tree(v[i],status,horizontal=False)

                status = 'Low Trim' if ((x_v%d)!=0 and len(v)==1) else 'Left Trim' if ((x_h%d)!=0 and (x_v%d)==0 and len(v)==1) else 'Main'
                search_tree(h[-1],status)
                a = tree.current.data

                status = 'Left Trim' if ((x_h%d)!=0 and h[0]==1 and len(v)==1) else 'Main'
                return transfer_matrix(a,v[-1],horizontal=False,status=status)

            for i in range(len(v)):
                status = 'Low Trim' if ((x_v%d)!=0 and i==0) else 'Left Trim' if ((x_h%d)!=0 and (x_v%d)==0 and i==0) else 'Main'
                search_tree(h[i],status)

                status = 'Left Trim' if ((x_h%d)!=0 and h[0]==1 and i==0) else 'Main'
                search_tree(v[i],status,horizontal=False)

            status = 'Low Trim' if ((x_v%d) != 0 and len(v)==0) else 'Left Trim' if ((x_h%d)!=0 and (x_v%d)==0 and len(v)==0) else 'Main'
            if len(h) == 1:
                return transfer_matrix(a,h[-1],status=status)
            a = tree.current.data
            return transfer_matrix(a,h[-1],status=status)

        if x == 0 and t == 0.:
            return a[1]

        x_h,x_v = math.ceil(t+x), math.floor(t+1-x)
        k = min(x_h,x_v)

        if (x_v%d) != 0:

            tile_set['Low Trim h_direct'] = np.loadtxt(f'{path_to_tileset}/h_direct_{d}x{x_v%d}.csv',dtype='complex_',delimiter=',')
            tile_set['Low Trim h_defect'] = np.loadtxt(f'{path_to_tileset}/h_defect_{d}x{x_v%d}.csv',dtype='complex_',delimiter=',')
            tile_set['Low Trim v_direct'] = np.loadtxt(f'{path_to_tileset}/v_direct_{d}x{x_v%d}.csv',dtype='complex_',delimiter=',')
            tile_set['Low Trim v_defect'] = np.loadtxt(f'{path_to_tileset}/v_defect_{d}x{x_v%d}.csv',dtype='complex_',delimiter=',')

            a = np.zeros(q**(2*(x_v%d)))
            a[q**(2*(x_v%d - 1))] = 1.0
            vertical_data = path_set((x_v//d + 1)-1)
        else:
            a = np.zeros(q**(2*d))
            a[q**(2*(d - 1))] = 1.0
            vertical_data = path_set((x_v//d)-1)
        
        if (x_h%d) != 0:

            tile_set['Left Trim h_direct'] = np.loadtxt(f'{path_to_tileset}/h_direct_{x_h%d}x{d}.csv',dtype='complex_',delimiter=',')
            tile_set['Left Trim h_defect'] = np.loadtxt(f'{path_to_tileset}/h_defect_{x_h%d}x{d}.csv',dtype='complex_',delimiter=',')
            tile_set['Left Trim v_direct'] = np.loadtxt(f'{path_to_tileset}/v_direct_{x_h%d}x{d}.csv',dtype='complex_',delimiter=',')
            tile_set['Left Trim v_defect'] = np.loadtxt(f'{path_to_tileset}/v_defect_{x_h%d}x{d}.csv',dtype='complex_',delimiter=',')

            horizontal_data = path_set((x_h//d + 1),k=k)
        else:
            horizontal_data = path_set((x_h//d),k=k)

        if (x_v%d) != 0 and (x_h%d) != 0:

            tile_set['Corner h_direct'] = np.loadtxt(f'{path_to_tileset}/h_direct_{x_h%d}x{x_v%d}.csv',dtype='complex_',delimiter=',')
            tile_set['Corner h_defect'] = np.loadtxt(f'{path_to_tileset}/h_defect_{x_h%d}x{x_v%d}.csv',dtype='complex_',delimiter=',')

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
    

    tile_set = dict()
    tile_set['Main h_direct'] = np.loadtxt(f'{path_to_tileset}/h_direct_{d}x{d}.csv',
                                          dtype='complex_',delimiter=',')
    tile_set['Main h_defect'] = np.loadtxt(f'{path_to_tileset}/h_defect_{d}x{d}.csv',
                                          dtype='complex_',delimiter=',')
    tile_set['Main v_direct'] = np.loadtxt(f'{path_to_tileset}/v_direct_{d}x{d}.csv',
                                          dtype='complex_',delimiter=',')
    tile_set['Main v_defect'] = np.loadtxt(f'{path_to_tileset}/v_defect_{d}x{d}.csv',
                                          dtype='complex_',delimiter=',')

    tree = Tree()
    df = pd.DataFrame()
    err = pd.Series()

    for T in range(2*tspan+1):

        t = float(T)/2

        if t == 0:
            s = pd.Series(np.array([1]),np.array([0]),name=t)        
            df = pd.concat([df, s.to_frame().T])
            err[t] = 1
            print(f'\ncomputed for T = {t}s')
            continue

        data = np.array([])
        inds = np.array([])

        for x in range(-T+1,T+1):
            x = float(x)/2
            inds = np.append(inds,x)
            data = np.append(data,[path_integral(x,t)])

        print(f'\ncomputed for T = {t}s')

        s = pd.Series(np.abs(data),inds,name=t)        
        df = pd.concat([df, s.to_frame().T])
        err[t] = np.abs(sum(data))

    df = df.reindex(sorted(df.columns,key=lambda num: float(num)), axis=1)
    df = df.fillna(0)
    df = df.iloc[::-1]
    print('\n',df,'\n')

    return df, err


def path_set(x:int,
             k:int=np.inf):  
    
    '''
        Generates a complete set of possible lists which can
        combine to form a complete set 
    '''

    def list_of_lists(x:int,
                      lists:np.ndarray = []):

        if x == 0:
            try:
                res[len(lists)].append(lists)
            except:
                res[len(lists)] = [lists]
            return
        elif len(lists) >= k:
            return 
        
        for i in range(1,x+1):
            sublist = lists.copy()
            sublist.append(i)
            list_of_lists(x-i,sublist)

    res = dict()
    list_of_lists(x)

    return res


def get_tiles(W:np.ndarray,
              d:int,
              path_to_tileset:str):

    '''
        gets the four types of tiles corresponding to horizontal defect,
        horizontal direct, vertical defect, vertical direct for a d dimensional
        case

        N.B. convention for index labelling is 'kx,y where (x,y) is the bond coordinate'
    '''

    if not os.path.isdir(path_to_tileset):
        os.mkdir(path_to_tileset)

    for d_v in range(1,d+1):
        for d_h in range(1,d+1):
            if (d_v != 1 and d_h != 1) or (d_v !=1 and d_h == 1):

                tensors = np.array([])
                
                if d_v != 1:
                    for i in range(d_h):
                        tensors = np.append(tensors,[qtn.Tensor(W[:,:,:,0],inds=(f'k{2*i+1},2',f'k{2*i+2},1',f'k{2*i},1'))])
                        tensors = np.append(tensors,[qtn.Tensor(W,inds=(f'k{2*i+1},{2*j+2}',f'k{2*i+2},{2*j+1}',
                                                                            f'k{2*i},{2*j+1}',f'k{2*i+1},{2*j}')) for j in range(1,d_v-1)])
                        tensors = np.append(tensors,[qtn.Tensor(W[0,:,:,:],inds=(f'k{2*i+2},{2*(d_v-1)+1}',f'k{2*i},{2*(d_v-1)+1}',f'k{2*i+1},{2*(d_v-1)}'))])
                else:
                    for i in range(d_h):
                        tensors = np.append(tensors,[qtn.Tensor(W[0,:,:,0],inds=(f'k{2*i+1},2',f'k{2*i+2},1',f'k{2*i},1'))])

                TN = qtn.TensorNetwork(tensors)
                val = TN.contract()

                reshape = tuple()
            
                for i in range(d_v):
                    reshape = reshape + (f'k{2*(d_h-1)+2},{2*i+1}',)
                for i in range(d_v):
                    reshape = reshape + (f'k0,{2*i+1}',)

                val = val.transpose(*reshape,inplace=True)
            
                h_direct = val.data
                h_direct = h_direct.reshape(-1,*h_direct.shape[-d_v:])
                h_direct = h_direct.reshape(*h_direct.shape[:1],-1)

                np.savetxt(f'{path_to_tileset}/h_direct_{d_h}x{d_v}.csv',h_direct,delimiter=',')

            else:
                h_direct = W[0,:,:,0]

                for i in range(d_h-1):
                    h_direct = np.einsum('ab,bc->ac',h_direct,W[0,:,:,0])

                np.savetxt(f'{path_to_tileset}/h_direct_{d_h}x{d_v}.csv',h_direct,delimiter=',')


    for d_v in range(1,d+1):
        for d_h in range(1,d+1):
            if d_v != 1 or d_h != 1:

                tensors = np.array([])

                tensors = np.append(tensors,[qtn.Tensor(W[:,:,:,0],inds=(f'k{2*i+1},2',f'k{2*i+2},1',f'k{2*i},1')) for i in range(d_h-1)])
                tensors = np.append(tensors,[qtn.Tensor(W[:,0,:,0],inds=(f'k{2*(d_h-1)+1},2',f'k{2*(d_h-1)},1'))])

                for j in range(1,d_v):
                    tensors = np.append(tensors,[qtn.Tensor(W,inds=(f'k{2*i+1},{2*j+2}',f'k{2*i+2},{2*j+1}',
                                                                    f'k{2*i},{2*j+1}',f'k{2*i+1},{2*j}')) for i in range(d_h-1)])
                    
                    tensors = np.append(tensors,[qtn.Tensor(W[:,0,:,:],inds=(f'k{2*(d_h-1)+1},{2*j+2}',f'k{2*(d_h-1)},{2*j+1}',f'k{2*(d_h-1)+1},{2*j}'))])

                TN = qtn.TensorNetwork(tensors)
                val = TN.contract()

                reshape = tuple()
                
                for i in range(d_h):
                    reshape = reshape + (f'k{2*i+1},{2*d_v}',)
                for i in range(d_v):
                    reshape = reshape + (f'k0,{2*i+1}',)

                val = val.transpose(*reshape,inplace=True)

                h_defect = val.data
                h_defect = h_defect.reshape(-1,*h_defect.shape[-d_v:])
                h_defect = h_defect.reshape(*h_defect.shape[:1],-1)

                np.savetxt(f'{path_to_tileset}/h_defect_{d_h}x{d_v}.csv',h_defect,delimiter=',')

            else:
                np.savetxt(f'{path_to_tileset}/h_defect_{d_h}x{d_v}.csv', W[:,0,:,0],delimiter=',')

    for d_v in range(1,d+1):
        for d_h in range(1,d+1):
            if (d_v != 1 and d_h != 1) or (d_v == 1 and d_h != 1):

                tensors = np.array([])
                    
                for j in range(d_v):
                    tensors = np.append(tensors,[qtn.Tensor(W[:,:,0,:],inds=(f'k1,{2*j+2}',f'k2,{2*j+1}',f'k1,{2*j}'))])
                    tensors = np.append(tensors,[qtn.Tensor(W,inds=(f'k{2*i+1},{2*j+2}',f'k{2*i+2},{2*j+1}',f'k{2*i},{2*j+1}',f'k{2*i+1},{2*j}')) for i in range(1,d_h-1)])
                    tensors = np.append(tensors,[qtn.Tensor(W[:,0,:,:],inds=(f'k{2*(d_h-1)+1},{2*j+2}',f'k{2*(d_h-1)},{2*j+1}',f'k{2*(d_h-1)+1},{2*j}'))])

                TN = qtn.TensorNetwork(tensors)
                val = TN.contract()

                reshape = tuple()
                
                for i in range(d_h):
                    reshape = reshape + (f'k{2*i+1},{2*(d_v-1)+2}',)
                for i in range(d_h):
                    reshape = reshape + (f'k{2*i+1},0',)

                val = val.transpose(*reshape,inplace=True)
                
                v_direct = val.data
                v_direct = v_direct.reshape(-1,*v_direct.shape[-d_h:])
                v_direct = v_direct.reshape(*v_direct.shape[:1],-1)

                np.savetxt(f'{path_to_tileset}/v_direct_{d_h}x{d_v}.csv',v_direct,delimiter=',')

            else:
                v_direct = W[:,0,0,:]
                for i in range(d_v-1):
                    v_direct = np.einsum('ab,bc->ac',v_direct,W[:,0,0,:])
                np.savetxt(f'{path_to_tileset}/v_direct_{d_h}x{d_v}.csv',v_direct,delimiter=',')

    for d_v in range(1,d+1):
        for d_h in range(1,d+1):
            if d_v != 1 or d_h != 1:

                tensors = np.array([])

                for j in range(d_v-1):              
                    tensors = np.append(tensors,[qtn.Tensor(W[:,:,0,:],inds=(f'k1,{2*j+2}',f'k2,{2*j+1}',f'k1,{2*j}'))])
                    tensors = np.append(tensors,[qtn.Tensor(W,inds=(f'k{2*i+1},{2*j+2}',f'k{2*i+2},{2*j+1}',
                                                                    f'k{2*i},{2*j+1}',f'k{2*i+1},{2*j}')) for i in range(1,d_h)])

                tensors = np.append(tensors,[qtn.Tensor(W[0,:,0,:],inds=(f'k2,{2*(d_v-1)+1}',f'k1,{2*(d_v-1)}'))])
                tensors = np.append(tensors,[qtn.Tensor(W[0,:,:,:],inds=(f'k{2*i+2},{2*(d_v-1)+1}',f'k{2*i},{2*(d_v-1)+1}',f'k{2*i+1},{2*(d_v-1)}')) for i in range(1,d_h)])

                TN = qtn.TensorNetwork(tensors)

                val = TN.contract()

                reshape = tuple()
                
                for i in range(d_v):
                    reshape = reshape + (f'k{2*d_h},{2*i+1}',)
                for i in range(d_h):
                    reshape = reshape + (f'k{2*i+1},0',)

                val = val.transpose(*reshape,inplace=True)
                
                v_defect = val.data
                v_defect = v_defect.reshape(-1,*v_defect.shape[-d_h:])
                v_defect = v_defect.reshape(*v_defect.shape[:1],-1)

                np.savetxt(f'{path_to_tileset}/v_defect_{d_h}x{d_v}.csv',v_defect,delimiter=',')

            else:
                np.savetxt(f'{path_to_tileset}/v_defect_{d_h}x{d_v}.csv',W[0,:,0,:],delimiter=',')







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