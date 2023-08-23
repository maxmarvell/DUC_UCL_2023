from cross_sections import fit
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams["font.family"] = "Times New Roman"

fig, ax = plt.subplots(1,5,sharey=True)
fig.suptitle("Spatial Distribution of Weight of Z(t) over Local Operators Z$_x$", fontweight ='bold',size=14, fontname="Times New Roman")

ax[0].set_ylabel("|<Z$_x$|Z(t)>|$^2$" ,size=14, fontname="Times New Roman", labelpad=10)

major_ticks = np.arange(-11.5,11.5,2)
minor_ticks = np.arange(-11.5,11.5,0.5)

for a in ax:
    a.minorticks_on()
    a.set_xticks(major_ticks)
    a.set_xticks(minor_ticks, minor=True)
    a.grid(which='minor', linewidth=0.5, alpha=0.5)
    a.set_xlabel("x", size=14, fontname="Times New Roman", labelpad=10)

pertubations = [7e-8,1e-7,3e-7,5e-7]
temperatures = [0.001,0.01,0.1,0.7]

bools = [True,True,True,True,True,True,True]
colours = ['b','c','g','r','m','y']


path = 'data/RandomTensorExact'

q = 2
e = 5e-7

# df = np.loadtxt(f'data/TensorExact/2_3806443768_3e-07.csv',delimiter=',',dtype='complex_',skiprows=1)
# tspan = len(df)
# c = 0

# for i in range(tspan-7,tspan,2):

#     cross_section = list(np.abs(df[i,:])**2)
#     space = list((np.array(range(len(cross_section)))  + 1 - (len(cross_section)/2))/2)
#     space_even = space[::2]
#     space_odd = space[1::2]
#     cross_section_even = cross_section[::2]
#     cross_section_odd = cross_section[1::2]

#     ax[0].scatter(space_odd, cross_section_odd, marker="d", color='k', s=5)

#     if bools[c]:
#         try:
#             dense_inputs_even, fit_data_even, R2_even = fit(space_even, cross_section_even)
#             dense_inputs_odd, fit_data_odd, R2_odd = fit(space_odd, cross_section_odd)
#             ax[0].plot(dense_inputs_odd, fit_data_odd, color=colours[c], linewidth=1.2, label=f"T = {float(i)/2}")
#         except: pass
#         #plt.plot(dense_inputs_odd, fit_data_odd, color=colours[i], linewidth=0.5, linestyle='dashed')
#     else:
#         ax[0].plot(space_odd, cross_section_odd, color=colours[c], linewidth=1.2, label=f"T = {float(i)/2}")
#         #plt.plot(space_odd, cross_section_odd, color=colours[i], linewidth=0.5, linestyle='dashed')

#     c += 1



for j in range(len(temperatures)):

    df = np.loadtxt(f'{path}/{q}_{e}_{temperatures[j]}.csv',delimiter=',',dtype='complex_',skiprows=1)
    tspan = len(df)

    c = 0

    for i in range(tspan-7,tspan,2):

        cross_section = list(np.abs(df[i,:])**2)
        space = list((np.array(range(len(cross_section)))  + 1 - (len(cross_section)/2))/2)
        space_even = space[::2]
        space_odd = space[1::2]
        cross_section_even = cross_section[::2]
        cross_section_odd = cross_section[1::2]

        ax[j].scatter(space_odd, cross_section_odd, marker="d", color='k', s=5)

        if bools[c]:
            try:
                dense_inputs_even, fit_data_even, R2_even = fit(space_even, cross_section_even)
                dense_inputs_odd, fit_data_odd, R2_odd = fit(space_odd, cross_section_odd)
                ax[j].plot(dense_inputs_odd, fit_data_odd, color=colours[c], linewidth=1.2, label=f"T = {float(i)/2}")
            except: pass
            #plt.plot(dense_inputs_odd, fit_data_odd, color=colours[i], linewidth=0.5, linestyle='dashed')
        else:
            ax[j].plot(space_odd, cross_section_odd, color=colours[c], linewidth=1.2, label=f"T = {float(i)/2}")
            #plt.plot(space_odd, cross_section_odd, color=colours[i], linewidth=0.5, linestyle='dashed')

        c += 1

plt.grid()
plt.legend(fontsize=12)
plt.show()        
        