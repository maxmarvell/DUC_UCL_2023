import quimb as qu
import quimb.tensor as qtn
import numpy as np

U = [1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1]

data = np.reshape(U, [2,2,2,2])

t00 = qtn.Tensor(data=data, inds=('k0,0','k0,1','00-01','00-10'), tags=('U0',))
t10 = qtn.Tensor(data=data, inds=('k1,0','00-01','01-01','01-10'), tags=('U1',))
t11 = qtn.Tensor(data=data, inds=('00-10','k1,1','10-01','10-10'), tags=('U2',))
# t20 = qtn.Tensor(data=data, inds=('k0,0','k0,1','00-01','00-10'), tags=('U0',))
# t21 = qtn.Tensor(data=data, inds=('k0,0','k0,1','00-01','00-10'), tags=('U0',))
# t22 = qtn.Tensor(data=data, inds=('k0,0','k0,1','00-01','00-10'), tags=('U0',))

tn = (t00 | t10 | t11)

tn.draw()