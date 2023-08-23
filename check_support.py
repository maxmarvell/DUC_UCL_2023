from P_generator import Exponential_Map
import quimb.tensor as qtn
import numpy as np

tensors = np.array([])

W = np.loadtxt(f'data/FoldedTensors/DU_2_3806443768.csv',
                delimiter=',',dtype='complex_')

P = np.loadtxt(f'data/FoldedPertubations/P_2_866021931.csv',
                delimiter=',',dtype='complex_')

q = 2
e = 5e-07

G = Exponential_Map(e,P)

PW = np.einsum('ab,bc->ac',G,W).reshape([q**2,q**2,q**2,q**2])

tensors = np.append(tensors,[qtn.Tensor(PW[:,0,:,:],inds=('l0','k1','u0'))])
tensors = np.append(tensors,[qtn.Tensor(PW,inds=(f'l{i}',f'k{i}',f'k{i+1}',f'u{i}')) for i in range(1,10)])
tensors = np.append(tensors,[qtn.Tensor(PW[:,:,0,:],inds=('l10',f'k10','u10'))])

TN = qtn.TensorNetwork(tensors)
TN.draw()

val = TN.contract()
data = val.data

U, S, Vt = np.linalg.svd(data)

print(Vt[1].shape)

I = np.array([1,0,0,0])
Z = np.array([0,1,0,0])

# eignevector = np.einsum('a,b,c,d,e,f,g->abcdefg',Z,I,I,I,I,Z,I)

# print(np.einsum('abcdefg,abcdefg->',eignevector,Vt[1]))

# mps = qtn.tensor_1d.MatrixProductState(tensors)