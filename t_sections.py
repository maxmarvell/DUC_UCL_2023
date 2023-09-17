import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


path = "./Larders/Larder_6/T24.0_d3_e10"
canvas = np.loadtxt(path,delimiter=',',dtype='complex_')
print(np.shape(canvas))
dim_t, dim_x = np.shape(canvas)
time = []
widths = []



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


t = -1
R2 = 1.0
while R2 >= 0.99:
    try:
        cross_section = list(np.abs(canvas[t,:])**2)
        space = list((np.array(range(len(cross_section)))  + 1 - (len(cross_section)/2))/2)
        cross_section = cross_section[::2]
        space = space[::2]
        dense_inputs, fit_data, R2, width = fit(space, cross_section)
        time.append((dim_t+t)/2)
        widths.append(width)
        t -= 1
    except:
        break

time = list(np.flip(np.array(time)))
widths = list(np.flip(np.array(widths)))



def power_fit(time, widths):

    def Function(x,a,p,b):
        return a*((x-time[0])**p) + b

    def partial_Function(a,p,b): 
        def P(x): return Function(x,a,p,b)
        return P

    dense_inputs = np.linspace(time[0],time[-1],100)
    opt_pars, cov_pars = curve_fit(Function, xdata=time, ydata=widths)
    astd = np.sqrt(float(np.diag(cov_pars)[0]))
    pstd = np.sqrt(float(np.diag(cov_pars)[1]))
    bstd = np.sqrt(float(np.diag(cov_pars)[2]))
    G = np.vectorize(partial_Function(opt_pars[0],opt_pars[1],opt_pars[2]))
    Gu = np.vectorize(partial_Function(opt_pars[0]+astd,opt_pars[1]+pstd,opt_pars[2]+bstd))
    Gl = np.vectorize(partial_Function(opt_pars[0]-astd,opt_pars[1]-pstd,opt_pars[2]-bstd))
    fit_data = G(dense_inputs)
    extreme_u = Gu(dense_inputs)
    extreme_l = Gl(dense_inputs)    

    R2 = r2_score(widths, G(time))

    return dense_inputs, fit_data, R2, opt_pars[1], extreme_u, extreme_l, pstd



fig, ax = plt.subplots()
plt.rcParams["font.family"] = "Times New Roman"
plt.xlabel("t", size=14, fontname="Times New Roman", labelpad=10)
plt.ylabel("$\sigma$" ,size=14, fontname="Times New Roman", labelpad=10)
plt.title("Width of Gaussian versus Time", fontweight ='bold',size=14, fontname="Times New Roman")


dense_inputs, fit_data, R2, p, extreme_u, extreme_l, pstd = power_fit(time, widths)


ax.minorticks_on()
major_ticks = np.arange(time[0],time[-1],1)
minor_ticks = np.arange(time[0],time[-1],0.25)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.grid(which='minor', linewidth=0.5, alpha=0.5)

plt.scatter(time, widths, marker="d", color='k', s=5)
plt.plot(dense_inputs, extreme_u, color='r', linewidth=0.3)
plt.plot(dense_inputs, extreme_l, color='r', linewidth=0.3)
plt.plot(dense_inputs, fit_data, color='k', linewidth=0.3, linestyle='dashed', label=f"p = {round(p,3)}±{round(pstd,3)}")
plt.fill_between(dense_inputs, extreme_l, extreme_u, color='k', alpha=0.1)
#plt.plot(time, widths, linewidth=0.7, linestyle='dashed', color='r')
plt.grid()
plt.legend()
plt.show()

print(time[0])
print(widths[0])

print(p)