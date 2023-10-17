from utils import *
import quimb.tensor as qtn
import numpy as np
import math

'''
    Code to compute benchmark exact correlations for DU circuit
    depth t and second point operator loaction x
'''

def contraction(x:float,
                t:float,
                q:int,
                W:np.ndarray,
                draw:bool = False):
    
    if x == 0 and t == 0:
        return 1
    
    Z, I = np.zeros([q**2]), np.zeros([q**2])
    Z[1], I[0] = 1, 1

    x_h, x_v = math.ceil(t+x), math.floor(t+1-x)

    DU = [[qtn.Tensor(

        W.reshape([q**2,q**2,q**2,q**2]),
        inds=(f'k{2*i+1},{2*j+2}',f'k{2*i+2},{2*j+1}',
                f'k{2*i},{2*j+1}',f'k{2*i+1},{2*j}'),
        tags=[f'UNITARY_{i},{j}'])

        for i in range(x_h)] for j in range(x_v)]

    a = [qtn.Tensor(Z,inds=('k0,1',),tags=['Z'])]
    
    if int(2*(t+x))%2 == 0:
        b = [qtn.Tensor(Z,inds=(f'k{2*x_h},{2*x_v-1}',),tags=['Z'])]
        e3 = [qtn.Tensor(I,inds=(f'k{2*x_h},{2*j+1}',),tags=['I']) for j in range(x_v-1)]
        e4 = [qtn.Tensor(I,inds=(f'k{2*j+1},{2*x_v}',),tags=['I']) for j in range(x_h)]
    else:
        b = [qtn.Tensor(Z,inds=(f'k{2*x_h-1},{2*x_v}',),tags=['Z'])]
        e3 = [qtn.Tensor(I,inds=(f'k{2*x_h},{2*j+1}',),tags=['I']) for j in range(x_v)]
        e4 = [qtn.Tensor(I,inds=(f'k{2*j+1},{2*x_v}',),tags=['I']) for j in range(x_h-1)]

    e1 = [qtn.Tensor(I,inds=(f'k{2*j+1},0',),tags=['I']) for j in range(x_h)]
    e2 = [qtn.Tensor(I,inds=(f'k0,{2*j+1}',),tags=['I']) for j in range(1,x_v)]

    TN = qtn.TensorNetwork((DU,a,b,e1,e2,e3,e4))

    if draw:

        fix = {
            'UNITARY_0,0': (0, 0),
            f'UNITARY_0,{x_v-1}': (0, 1),
            f'UNITARY_{x_h-1},0': (1, 0),
            f'UNITARY_{x_h-1},{x_v-1}': (1, 1),
        }

        if x_h > 1 and x_v > 1:
            TN.draw(fix=fix)
        else:
            TN.draw()

    return TN.contract()

def contraction_random(x:float,
                       t:float,
                       q:int,
                       PW:dict,
                       gates:np.ndarray,
                       temp:float,
                       draw:bool = False):
    
    if x == 0 and t == 0:
        return 1

    Z, I = np.zeros([q**2]), np.zeros([q**2])
    Z[1], I[0] = 1, 1

    x_h, x_v = math.ceil(t+x), math.floor(t+1-x)

    DU = [[qtn.Tensor(

        PW[distribute(gates,temp)],
        inds=(f'k{2*i+1},{2*j}',f'k{2*i},{2*j+1}',
                f'k{2*i+2},{2*j+1}',f'k{2*i+1},{2*j+2}'),
        tags=[f'UNITARY_{i},{j}'])

        for i in range(x_h)] for j in range(x_v)]

    a = [qtn.Tensor(Z,inds=('k0,1',),tags=['Z'])]
    
    if int(2*(t+x))%2 == 0:
        b = [qtn.Tensor(Z,inds=(f'k{2*x_h},{2*x_v-1}',),tags=['Z'])]
        e3 = [qtn.Tensor(I,inds=(f'k{2*x_h},{2*j+1}',),tags=['I']) for j in range(x_v-1)]
        e4 = [qtn.Tensor(I,inds=(f'k{2*j+1},{2*x_v}',),tags=['I']) for j in range(x_h)]
    else:
        b = [qtn.Tensor(Z,inds=(f'k{2*x_h-1},{2*x_v}',),tags=['Z'])]
        e3 = [qtn.Tensor(I,inds=(f'k{2*x_h},{2*j+1}',),tags=['I']) for j in range(x_v)]
        e4 = [qtn.Tensor(I,inds=(f'k{2*j+1},{2*x_v}',),tags=['I']) for j in range(x_h-1)]

    e1 = [qtn.Tensor(I,inds=(f'k{2*j+1},0',),tags=['I']) for j in range(x_h)]
    e2 = [qtn.Tensor(I,inds=(f'k0,{2*j+1}',),tags=['I']) for j in range(1,x_v)]

    TN = qtn.TensorNetwork((DU,a,b,e1,e2,e3,e4))

    if draw:

        fix = {
            'UNITARY_0,0': (0, 0),
            f'UNITARY_0,{x_v-1}': (0, 1),
            f'UNITARY_{x_h-1},0': (1, 0),
            f'UNITARY_{x_h-1},{x_v-1}': (1, 1),
        }

        if x_h > 1 and x_v > 1:
            TN.draw(fix=fix)
        else:
            TN.draw()

    return TN.contract()