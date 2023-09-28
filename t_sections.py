import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


path = "./Larders/Larder_0/T24.0_d3_e5"
canvas = np.loadtxt(path,delimiter=',')
print(np.shape(canvas))
dim_t, dim_x = np.shape(canvas)
T = float((dim_t-1))/2
time = []
widths = []
errors = []



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
    cstd = np.sqrt(float(np.diag(cov_pars)[2]))

    R2 = r2_score(cross_section, G(space))

    return dense_inputs, fit_data, R2, opt_pars[2], cstd


t = -1
R2 = 1.0
while R2 >= 0.99:
    try:
        cross_section = list(canvas[t,:])
        space = np.arange(-(T-0.5),(T+0.5),0.5)
        if (dim_t%2 == 0):
            offset = int((1+((dim_t+t)%2))%2)
        else:
            offset = int((dim_t+t)%2)
        cross_section = cross_section[offset::2]
        space = space[offset::2]
        dense_inputs, fit_data, R2, width, err = fit(space, cross_section)
        time.append((dim_t+t)/2)
        widths.append(width)
        errors.append(err)
        t -= 1
    except:
        break

time = list(np.flip(np.array(time)))
widths = list(np.flip(np.array(widths)))
errors = list(np.flip(np.array(errors)))



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
major_ticks = np.arange(time[0],time[-1],2)
minor_ticks = np.arange(time[0],time[-1],0.25)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.grid(which='minor', linewidth=0.5, alpha=0.5)

plt.errorbar(time,widths,yerr=errors,marker="d",color='k',ms=2,linewidth=0.2,linestyle="dashed",elinewidth=0.5,capsize=2)

plt.plot(dense_inputs, fit_data, color='r', linewidth=0.5, linestyle='dashed', label=f"p = {round(p,3)}±{round(pstd,3)}")

#plt.plot(time, widths, linewidth=0.7, linestyle='dashed', color='r')
plt.grid()
plt.legend()
plt.show()

