import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


path = "./Larders/Larder_3/T24.0_d3_e12"
canvas = np.loadtxt(path,delimiter=',')
dim_t, dim_x = np.shape(canvas)
T = 23.5
time = []
widths = []
plt.imshow(canvas,cmap='hot', interpolation='nearest')
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





fig, ax = plt.subplots()
plt.rcParams["font.family"] = "Times New Roman"
plt.xlabel("x", size=14, fontname="Times New Roman", labelpad=10)
plt.ylabel("<Z$_x$|Z(t)>" ,size=14, fontname="Times New Roman", labelpad=10)
plt.title(f"Diffusion of Central Gaussian Lump at Late Times", fontweight ='bold',size=14, fontname="Times New Roman")

t = -1
R2 = 1.0
while R2 >= 0.99:
        try:
            space = np.arange(-(T-0.5),(T+0.5),0.5) 
            if (dim_t%2 == 0):
                    offset = int((1+((dim_t+t)%2))%2)
            else:
                offset = int((dim_t+t)%2)
            cross_section = list(canvas[t,:])
            space = space[offset::2]
            cross_section = cross_section[offset::2]
            dense_inputs, fit_data, R2, width = fit(space, cross_section)
            if not R2 >= 0.99:
                break
            time.append((dim_t+t)/2)
            widths.append(width)
            if t == -1:
                plt.scatter(space, cross_section, marker="d", color='k', s=5)
                plt.plot(dense_inputs, fit_data, color='b', linewidth=0.6, label=f"t = {T}")
            else:     
                plt.scatter(space, cross_section, marker="d", color='k', s=5)
                plt.plot(dense_inputs, fit_data, color='r', linewidth=0.3, linestyle='dashed')
            t -= 1          
        except:
            break

plt.scatter(space, cross_section, marker="d", color='k', s=5)
plt.plot(dense_inputs, fit_data, color='g', linewidth=0.6, label=f"t = {float(dim_t+t)/2}")

ax.minorticks_on()
major_ticks = np.arange(space[0],space[-1],3)
minor_ticks = np.arange(space[0],space[-1],0.25)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.grid(which='minor', linewidth=0.5, alpha=0.5)
plt.grid()
plt.legend()
plt.show()
