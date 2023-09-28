import numpy as np
import quimb.tensor as qtn
from scipy.linalg import expm
from os import mkdir


def main():

    q = 2
    d = 3
    
    for i in [0,1,2,3,4,5]:
        W_path = f"./Sample_Tensor_{i}.csv"
        P_path = f"./Sample_Perturbation_{i}.csv"
        W = np.loadtxt(W_path,delimiter=',')
        G = np.loadtxt(P_path,delimiter=',')
        G /= np.linalg.norm(G)

        range = [5,5.5,6,6.5,7,
             7.5,8,8.5,9,9.5,
             10,10.5,11,11.5,12,
             12.5,13,13.5,14,14.5]
        print("\n")

        for e in range:
            P = expm(e*G)
            PW = np.einsum('ab,bc->ac',P,W).reshape([q**2,q**2,q**2,q**2])
            mkdir(f"./Tile_Zoo_{i}_e{e}")
            dir = f"./Tile_Zoo_{i}_e{e}"
            get_tiles(PW, d, dir)





def get_tiles(W:np.ndarray,
              d:int,
              dir:str):

    '''
        gets the four types of tiles corresponding to horizontal defect,
        horizontal direct, vertical defect, vertical direct for a d dimensional
        case

        N.B. convention for index labelling is "kx,y where (x,y) is the bond coordinate"
    '''

    #tiles = dict()



    # HORIZONTAL DIRECT TILE

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

                np.savetxt(f"{dir}/h_direct_{d_h}x{d_v}", h_direct, delimiter=",")

            else:
                h_direct = W[0,:,:,0]
                for i in range(d_h-1):
                    h_direct = np.einsum('ab,bc->ac',h_direct,W[0,:,:,0])
                np.savetxt(f"{dir}/h_direct_{d_h}x{d_v}", h_direct, delimiter=",")



    # HORIZONTAL DEFECT TILE

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

                np.savetxt(f"{dir}/h_defect_{d_h}x{d_v}", h_defect, delimiter=",")

            else:
                np.savetxt(f"{dir}/h_defect_{d_h}x{d_v}", W[:,0,:,0], delimiter=",")



    ### VERTICAL DIRECT TILE

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

                np.savetxt(f"{dir}/v_direct_{d_h}x{d_v}", v_direct, delimiter=",")

            else:
                v_direct = W[:,0,0,:]
                for i in range(d_v-1):
                    v_direct = np.einsum('ab,bc->ac',v_direct,W[:,0,0,:])
                np.savetxt(f"{dir}/v_direct_{d_h}x{d_v}", v_direct, delimiter=",")



    ### VERTICAL DEFECT TILE

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

                np.savetxt(f"{dir}/v_defect_{d_h}x{d_v}", v_defect, delimiter=",")

            else:
                np.savetxt(f"{dir}/v_defect_{d_h}x{d_v}", W[0,:,0,:], delimiter=",")

   


if __name__ == '__main__':
    main()