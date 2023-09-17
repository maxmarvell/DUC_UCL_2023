import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


def fit(space, cross_section):

    def Gaussian(x,a,b,c):
        return a*np.exp(-((x-b)**2)/(2*c**2))

    def partial_gauss(a,b,c): 
        def P(x): return Gaussian(x,a,b,c)
        return P

    dense_inputs = np.linspace(space[0],space[-1],100)
    opt_pars, cov_pars = curve_fit(Gaussian, xdata=space, ydata=cross_section)
    G = np.vectorize(partial_gauss(opt_pars[0],opt_pars[1],opt_pars[2]))
    fit_data = G(dense_inputs)

    R2 = r2_score(cross_section, G(space))

    return dense_inputs, fit_data, R2, opt_pars[2]


#plt.imshow(np.abs(canvas),cmap='hot', interpolation='nearest')
#plt.show()

fig, ax = plt.subplots()
plt.rcParams["font.family"] = "Times New Roman"
plt.xlabel("x", size=14, fontname="Times New Roman", labelpad=10)
plt.ylabel("|<Z$_x$|Z(t)>|$^2$" ,size=14, fontname="Times New Roman", labelpad=10)
plt.title("Spatial Distribution of Weight of Z(t) over Local Operators Z$_x$", fontweight ='bold',size=14, fontname="Times New Roman")

path = "./Larders/Larder_5/T24.0_d3_e7"
canvas = np.loadtxt(path,delimiter=',',dtype='complex_')
dim_t, dim_x = np.shape(canvas)
space = list((np.array(range(dim_x))  + 1 - (dim_x/2))/2)

ax.minorticks_on()
major_ticks = np.arange(space[0],space[-1],2)
minor_ticks = np.arange(space[0],space[-1],0.5)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.grid(which='minor', linewidth=0.5, alpha=0.5)

bools = [False, False, False, False, False, False, False, False, False, False, False]
colours = ['b','c','g','r','m','k','k','k','k','k','k']
times = [8,10,12,14,16,18,20,22,24,26,28]


for i in range(11):
    cross_section = list(np.abs(canvas[times[i],:])**2)
    space_even = space[1::2]
    space_odd = space[::2]
    cross_section_even = cross_section[1::2]
    cross_section_odd = cross_section[::2]

    plt.scatter(space_even, cross_section_even, marker="d", color='k', s=5)

    if bools[i]:
        dense_inputs_even, fit_data_even, R2_even, width_even = fit(space_even, cross_section_even)
        dense_inputs_odd, fit_data_odd, R2_odd, width_odd = fit(space_odd, cross_section_odd)
        plt.plot(dense_inputs_even, fit_data_even, color=colours[i], linewidth=1.2, label=f"t = {float(times[i])/2}")
        #plt.plot(dense_inputs_odd, fit_data_odd, color=colours[i], linewidth=0.5, linestyle='dashed')
        print(R2_even)
    else:
        plt.plot(space_even, cross_section_even, color=colours[i], linewidth=1.2, label=f"t = {float(times[i])/2}")
        #plt.plot(space_odd, cross_section_odd, color=colours[i], linewidth=0.5, linestyle='dashed')


plt.grid()
plt.legend(fontsize=12)
plt.show()
