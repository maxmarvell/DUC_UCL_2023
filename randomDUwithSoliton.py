#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 18:01:36 2023

@author: Tom
"""
 
import numpy as np
from scipy.stats import unitary_group
from scipy.optimize import minimize

def real_to_complex(z):      # real vector of length 2n -> complex of length n
    return z[:len(z)//2] + 1j * z[len(z)//2:]
def complex_to_real(z):      # complex vector of length n -> real of length 2n
    return np.concatenate((np.real(z), np.imag(z)))

def finddUconstrained(q, soliton = False, print_result = False):
    """This function maximises tr(UR),  where U is the unitary we are optimising
    and R is some random q**2 by q**2 matrix. We have included ||UU^{\dag} - I|| (unitarity), 
    ||\tilde{U}tilde{U}^{\dag} - I|| (dual-unitarity) as constraints to be minimised. 
    Optionally, another constraint ||UZ_0 - Z_1U|| + ||UZ_1 - UZ_0||, which enforces 
    that a generalised Z operator is a right- and left-moving soliton, can be added.
    """
    U_0 = unitary_group.rvs(q**2)
    R = np.random.rand(q**2, q**2)


    def unitarity(x):
        q = int(np.sqrt(np.sqrt(len(x))))
        U = x.reshape([q,q,q,q])
        IdId = np.einsum("ac,bd->abcd", np.identity(q), np.identity(q))
        return np.linalg.norm(np.einsum('abcd,efcd -> abef', U, np.conj(U))-IdId)
    def dual_unitarity(x):
        q = int(np.sqrt(np.sqrt(len(x))))
        U = x.reshape([q,q,q,q])
        IdId = np.einsum("ac,bd->abcd", np.identity(q), np.identity(q))
        return np.linalg.norm(np.einsum('fbea,fdec->abcd',np.conj(U),U)-IdId)
    def randomise(x, R):
        q = int(np.sqrt(np.sqrt(len(x))))
        U = x.reshape([q**2, q**2])
        return -abs(np.trace(np.einsum("ab,bc->ac", U, R)))
    
    def soliton(x, Z):
        q = int(np.sqrt(np.sqrt(len(x))))
        U = x.reshape([q,q,q,q])
        Z_0 = np.einsum('ac,bd->abcd', Z, np.identity(q))
        Z_1 = np.einsum('ac,bd->abcd', np.identity(q), Z)
        return np.linalg.norm(
            np.einsum('abcd,cdef -> abef', U, Z_0) \
                              - np.einsum('abcd,cdef -> abef', Z_1, U)) \
            + np.linalg.norm(np.einsum('abcd,cdef -> abef', U, Z_1) \
                             - np.einsum('abcd,cdef -> abef', Z_0, U)
                             )
    
    
    if not soliton:
        
        cons = [{'type':'eq', 'fun': lambda z:  unitarity(real_to_complex(z))}, 
                {'type':'eq', 'fun': lambda z: dual_unitarity(real_to_complex(z))}]

        
    if soliton:
        
        Z = np.zeros([q,q])
        Z[0,0] = 1
        Z[1,1] = -1
        
        cons = [{'type':'eq', 'fun': lambda z:  unitarity(real_to_complex(z))}, 
                {'type':'eq', 'fun': lambda z: dual_unitarity(real_to_complex(z))}, 
                {'type':'eq', 'fun': lambda z: soliton(real_to_complex(z), Z)}]
    
    result = minimize(
        lambda z: randomise(real_to_complex(z), R),  \
                      x0 = complex_to_real(U_0.flatten()), \
                      method='SLSQP', constraints=cons, options={'maxiter': 1000}
                      )
    
    if print_result == True:
        print(result)
        
    return real_to_complex(result.x)