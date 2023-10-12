import numpy as np
import matplotlib.pyplot as plt
from utils import fit

path = "./Larders/Larder_5/T24.0_d3_e7"
canvas = np.loadtxt(path,delimiter=',',dtype='complex_')
T = 23.5
space = np.arange(-(T-0.5),(T+0.5),0.5)


fig, ax = plt.subplots()
plt.rcParams["font.family"] = "Times New Roman"
plt.xlabel("x", size=14, fontname="Times New Roman", labelpad=10)
plt.ylabel("|<Z$_x$|Z(t)>|$^2$" ,size=14, fontname="Times New Roman", labelpad=10)
plt.title("Spatial Distribution of Weight of Z(t) over Local Operators Z$_x$", fontweight ='bold',size=14, fontname="Times New Roman")
#plt.text(13.0,0.005,"Integer Lattice", size=12)
ax.minorticks_on()
major_ticks = np.arange(space[0],space[-1]+0.5,4)
minor_ticks = np.arange(space[0],space[-1]+0.5,0.5)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.grid(which='minor', linewidth=0.5, alpha=0.5)


bools = [True, True, True, True, True, True]
colours = ['r','m','b','g','c','k']
times = [6.0,6.5,8.0,12.0,23.5]
parity = "odd"
print(list(np.flip(np.abs(canvas[0,:])**2)).index(1.0))
print(len(space))

for i in range(5):

    if parity == "even":
        if (int(2*T)%2 == 0):
            offset = int((1+((2*times[i])%2))%2)
        else:
            offset = int((2*times[i])%2)
    else:
        if (int(2*T)%2 == 0):
            offset = int((2*times[i])%2)
        else:
            offset = int((1+((2*times[i])%2))%2)
           
    space = np.arange(-(T-0.5),(T+0.5),0.5)
    cross_section = list(np.flip(np.abs(canvas[int(2*times[i]),:])**2))
    space = space[offset::2]
    cross_section = cross_section[offset::2]


    if bools[i]:
        dense_inputs, fit_data, R2, width = fit(space, cross_section)
        plt.plot(dense_inputs, fit_data, color=colours[i],linewidth=0.5,label=f"t = {times[i]}")
    else:
        plt.plot(space, cross_section, color=colours[i],linewidth=0.5,label=f"t = {times[i]}")

    plt.scatter(space, cross_section, marker="d", color='k', s=4)

plt.grid()
plt.legend(fontsize=12)
plt.show()
