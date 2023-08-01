import quimb as qu
import quimb.tensor as qtn
import numpy as np

w = np.loadtxt('./data/FoldedTensors/DU_2_3367707468166924861.csv',
               delimiter=',', dtype='complex_')

z,i = np.zeros([4]),np.zeros([4])
z[1],i[0] = 1,1

Z = qtn.Tensor(z,inds=('a0',),tags=['PAULI','Z'])
I = qtn.Tensor(i,inds=('a1',),tags=['PAULI','I'])
W = qtn.Tensor(w.reshape([4,4,4,4]),inds=('a0','a1','b0','b1'),tags=['UNITARY'])

x_h, x_v = 4, 4

tnsrs = [[qtn.Tensor(w.reshape([4,4,4,4]),inds=('a','b','c','d'),tags=[f'UNITARY_{i}{j}']) \
           for i in range(x_h)] for j in range(x_v)]

ab = [qtn.Tensor(z,inds=('k01',),tags=['Z']),qtn.Tensor(z,inds=('k87',),tags=['Z'])]
e1 = [qtn.Tensor(i,inds=(f'k0{2*j+1}',),tags=['I']) for j in range(1,x_h)]
e2 = [qtn.Tensor(i,inds=(f'k{2*j+1}0',),tags=['I']) for j in range(x_v)]
e3 = [qtn.Tensor(i,inds=(f'k8{2*j+1}',),tags=['I']) for j in range(x_h-1)]
e4 = [qtn.Tensor(i,inds=(f'k{2*j+1}8',),tags=['I']) for j in range(x_v)]

for i in range(x_h):
    for j in range(x_v):
        index_map = {'a':f'k{2*i}{2*j+1}','b':f'k{2*i+1}{2*j}','c':f'k{2*i+2}{2*j+1}','d':f'k{2*i+1}{2*j+2}'}
        tnsrs[i][j].reindex(index_map,inplace=True)

fix = {
    'UNITARY_00': (0, 0),
    f'UNITARY_0{x_v}': (0, 1),
    f'UNITARY_{x_h}0': (1, 0),
    f'UNITARY_{x_h}{x_v}': (1, 1),
}

TN = qtn.TensorNetwork((tnsrs,ab,e1,e2,e3,e4))

print(np.abs(TN.contract()))

TN.draw(fix=fix)
