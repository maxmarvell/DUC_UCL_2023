import quimb as qu
import quimb.tensor as qtn
import numpy as np
from time import time
import math

"""
    Code to compute benchmark exact correlations for DU circuit
    depth t and second point operator loaction x
"""

def main():

    W = np.loadtxt('./data/FoldedTensors/DU_2_3367707468166924861.csv',
               delimiter=',', dtype='complex_')
    
    exact_contraction(4,4,2,W)
    

def exact_contraction(x:float,
                      t:int,
                      q:int,
                      W:np.ndarray):

    Z, I = np.zeros([q**2]), np.zeros([q**2])
    Z[1], I[0] = 1, 1

    x_h, x_v = t + math.ceil(x), t + 1 - math.ceil(x)

    print(x_h,x_v)

    DU = [[qtn.Tensor(W.reshape([q**2,q**2,q**2,q**2]),inds=('a','b','c','d'),
                        tags=[f'UNITARY_{i}{j}']) for i in range(x_h)] for j in range(x_v)]

    a, b = [qtn.Tensor(Z,inds=('k01',),tags=['Z'])],[qtn.Tensor(Z,inds=(f'k{2*x_h-1}{2*x_v}',),tags=['Z'])]

    e1 = [qtn.Tensor(I,inds=(f'k{2*j+1}0',),tags=['I']) for j in range(x_h)]
    e2 = [qtn.Tensor(I,inds=(f'k0{2*j+1}',),tags=['I']) for j in range(1,x_v)]
    e3 = [qtn.Tensor(I,inds=(f'k{2*x_h}{2*j+1}',),tags=['I']) for j in range(x_v)]
    e4 = [qtn.Tensor(I,inds=(f'k{2*j+1}{2*x_v}',),tags=['I']) for j in range(x_h-1)]

    for i in range(x_h):
        for j in range(x_v):
            index_map = {'a':f'k{2*i}{2*j+1}',
                         'b':f'k{2*i+1}{2*j}',
                         'c':f'k{2*i+2}{2*j+1}',
                         'd':f'k{2*i+1}{2*j+2}'}
            DU[j][i].reindex(index_map,inplace=True)

    TN = qtn.TensorNetwork((DU,a,b,e1,e2,e3,e4))


    print(TN)

    TN.draw(show_tags=True)

    

    pass

# e5 = [qtn.Tensor(i,inds=(f'k0{2*j+1}',),tags=['I']) for j in range(x_h)]
# e6 = [qtn.Tensor(i,inds=(f'k2{2*j+1}',),tags=['I']) for j in range(x_h)]
# e7 = [qtn.Tensor(z,inds=('k10',),tags=['Z']),qtn.Tensor(z,inds=(f'k1{2*x_h}',),tags=['Z'])]

# fix = {
#     'UNITARY_00': (0, 0),
#     f'UNITARY_0{x_v}': (0, 1),
#     f'UNITARY_{x_h}0': (1, 0),
#     f'UNITARY_{x_h}{x_v}': (1, 1),
# }

# print(np.abs(TN.contract()))

# TN.draw(fix=fix)


# skeleton = [qtn.Tensor(w.reshape([4,4,4,4]),inds=('a','b','c','d'),tags=[f'UNITARY_{i}']) \
#            for i in range(x_h)]

# for i in range(x_h):
#     index_map = {'a':f'k1{2*i}','b':f'k0{2*i+1}','c':f'k2{2*i+1}','d':f'k1{2*i+2}'}
#     skeleton[i].reindex(index_map,inplace=True)


# TN2 = qtn.TensorNetwork((skeleton,e5,e6,e7))

# print(np.abs(TN2.contract()))

# TN2.draw()

if __name__ == "__main__":
    main()