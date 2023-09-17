import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


path = "./Larders/Larder_6/T24.0_d3_e11"
canvas = np.loadtxt(path,delimiter=',',dtype='complex_')
print(np.shape(canvas))
dim_t, dim_x = np.shape(canvas)
T = 24.0
e = 0.0000001
time = []
widths = []
plt.imshow(np.abs(canvas),cmap='hot', interpolation='nearest')
plt.show()



def fit(space, cross_section):

    def Gaussian(x,a,b,c):
        return a*np.exp(-((x-b)**2)/(2*c**2))

    def partial_gauss(a,b,c): 
        def P(x): return Gaussian(x,a,b,c)
        return P

    dense_inputs = np.linspace(space[0],space[-1],1000)
    opt_pars, cov_pars = curve_fit(Gaussian, xdata=space, ydata=cross_section)
    G = np.vectorize(partial_gauss(opt_pars[0],opt_pars[1],opt_pars[2]))
    fit_data = G(dense_inputs)

    R2 = r2_score(cross_section, G(space))

    return dense_inputs, fit_data, R2, opt_pars[2]



space = list((np.array(range(dim_x))  + 1 - (dim_x/2))/2)
space = space[::2]
fig, ax = plt.subplots()
plt.rcParams["font.family"] = "Times New Roman"
plt.xlabel("x", size=14, fontname="Times New Roman", labelpad=10)
plt.ylabel("|<Z$_x$|Z(t)>|$^2$" ,size=14, fontname="Times New Roman", labelpad=10)
plt.title(f"Spatial Distribution of Weight of Z(t) over Local Operators Z$_x$ \n $\epsilon$ = {e}", fontweight ='bold',size=14, fontname="Times New Roman")

'''''
t = -45

cross_section = list(np.abs(canvas[t,:]))
cross_section = cross_section[::2]
dense_inputs, fit_data, R2, width = fit(space, cross_section)
time.append((dim_t+t)/2)
widths.append(width)
plt.scatter(space, cross_section, marker="d", color='k', s=5)
plt.plot(dense_inputs, fit_data, color='r', linewidth=0.3, linestyle='dashed', label=f"t = {T + 1 + t/2}")
print(R2)

'''''
t = -1
R2 = 1.0
while R2 >= 0.99:
        try:
            cross_section = list(np.abs(canvas[t,:])**2)
            cross_section = cross_section[::2]
            dense_inputs, fit_data, R2, width = fit(space, cross_section)
            if not R2 >= 0.99:
                break
            time.append((dim_t+t)/2)
            widths.append(width)
            plt.scatter(space, cross_section, marker="d", color='k', s=5)
            plt.plot(dense_inputs, fit_data, color='r', linewidth=0.3, linestyle='dashed', label=f"t = {T + (1+t)/2}")
            t -= 2           
        except:
            break


ax.minorticks_on()
major_ticks = np.arange(space[0],space[-1],2)
minor_ticks = np.arange(space[0],space[-1],0.25)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.grid(which='minor', linewidth=0.5, alpha=0.5)
plt.grid()
plt.legend()
plt.show()
