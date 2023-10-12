from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.linalg import expm 
import numpy as np

def list_generator(x:int,
                   data:dict,
                   k:int=np.inf,
                   lists:np.ndarray=[]):
    '''
        Generates a complete set of possible lists which can
        combine to form a complete set 
    '''

    if x == 0:
        try:
            data[len(lists)].append(lists)
        except:
            data[len(lists)] = [lists]
        return
    elif len(lists) >= k:
        return 

    for i in range(1,x+1):
        sublist = lists.copy()
        sublist.append(i)
        list_generator(x-i,data,k,sublist)

def randomise(P, R):
    return -abs(np.einsum("a,a->", P, R))

def Exponential_Map(e, P):
    return expm(e*P)

def fit(space, cross_section):

    def Gaussian(x,a,b,c):
        return a*np.exp(-((x-b)**2)/(2*c**2))

    def partial_gauss(a,b,c): 
        def P(x): return Gaussian(x,a,b,c)
        return P

    dense_inputs = np.linspace(space[0],space[-1],100)
    opt_pars, _ = curve_fit(Gaussian, xdata=space, ydata=cross_section)
    G = np.vectorize(partial_gauss(opt_pars[0],opt_pars[1],opt_pars[2]))
    fit_data = G(dense_inputs)

    R2 = r2_score(cross_section, G(space))

    return dense_inputs, fit_data, R2, opt_pars[2]