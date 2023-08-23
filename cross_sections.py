import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def main():

    #plt.imshow(np.abs(canvas),cmap='hot', interpolation='nearest')
    #plt.show()

    fig, ax = plt.subplots()
    plt.rcParams["font.family"] = "Times New Roman"
    plt.xlabel("x", size=14, fontname="Times New Roman", labelpad=10)
    plt.ylabel("|<Z$_x$|Z(t)>|$^2$" ,size=14, fontname="Times New Roman", labelpad=10)
    plt.title("Spatial Distribution of Weight of Z(t) over Local Operators Z$_x$", fontweight ='bold',size=14, fontname="Times New Roman")

    ax.minorticks_on()
    major_ticks = np.arange(-11.5,11.5,2)
    minor_ticks = np.arange(-11.5,11.5,0.5)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.grid(which='minor', linewidth=0.5, alpha=0.5)

    # ax.set_ylim(0.0, 0.0007)
    # ax.set_xlim(-11.0,11.0)

    bools = [False,False,False,False,False,False,False]
    colours = ['b','c','g','r', 'm','k']
    temperature = [0.5,0.9,1.1,1.5]
    e = [5e-8,9e-8,1e-7,3e-7,5e-7]

    path = f"data/RandomPathIntegral0.7/2_1e-07_0.7w.csv"
    canvas = np.loadtxt(path,delimiter=',',dtype='complex_',skiprows=1)

    c = 0

    for i in range(14,len(canvas),2):
        # path = f"data/RandomTensorExact/2_3e-07_0.9.csv"
        # canvas = np.loadtxt(path,delimiter=',',dtype='complex_',skiprows=1)
        cross_section = list(np.abs(canvas[i,:])**2)
        space = list((np.array(range(len(cross_section)))  + 1 - (len(cross_section)/2))/2)
        space_even = space[::2]
        space_odd = space[1::2]
        cross_section_even = cross_section[::2]
        cross_section_odd = cross_section[1::2]

        plt.scatter(space_odd, cross_section_odd, marker="d", color='k', s=5)

        if bools[c]:
            dense_inputs_even, fit_data_even, R2_even = fit(space_even, cross_section_even)
            dense_inputs_odd, fit_data_odd, R2_odd = fit(space_odd, cross_section_odd)
            plt.plot(dense_inputs_odd, fit_data_odd, color=colours[c], linewidth=1.2, label=f"T = {float(i)/2}")
            #plt.plot(dense_inputs_odd, fit_data_odd, color=colours[i], linewidth=0.5, linestyle='dashed')
        else:
            plt.plot(space_odd, cross_section_odd, color=colours[c], linewidth=1.2, label=f"T = {float(i)/2}")
            #plt.plot(space_odd, cross_section_odd, color=colours[i], linewidth=0.5, linestyle='dashed')

        c += 1


    plt.grid()
    plt.legend(fontsize=12)
    plt.show()


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

    return dense_inputs, fit_data, R2

if __name__ =='__main__':
    main()