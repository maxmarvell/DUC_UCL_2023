import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


path = "./Charge_Scores.csv"
data = pd.read_csv(path)
charges1k = np.abs(data["3,3"].apply(lambda x: np.complex(x)) - 1)
charges2k = np.abs(data["3,6"].apply(lambda x: np.complex(x)) - 1)
charges3k = np.abs(data["3,12"].apply(lambda x: np.complex(x)) - 1)
charges4k = np.abs(data["3,k"].apply(lambda x: np.complex(x)) - 1) 
time = range(48)


fig, ax = plt.subplots()
plt.rcParams["font.family"] = "Times New Roman"
plt.xlabel("t", size=14, fontname="Times New Roman", labelpad=10)
plt.ylabel("|<Q> - 1|" ,size=14, fontname="Times New Roman", labelpad=10)
plt.title("Accumulated Error in Charge Conservation up to t$^{th}$ Layer", fontweight ='bold',size=14, fontname="Times New Roman")
ax.minorticks_on()
major_ticks = np.arange(time[0],time[-1],10)
minor_ticks = np.arange(time[0],time[-1],0.5)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.grid(which='minor', linewidth=0.5, alpha=0.5)

plt.plot(time,charges1k,marker='d',color='k',ms=2,linewidth=0.8,label="n$_{max}$ = 3")
plt.plot(time,charges2k,marker='d',color='r',ms=2,linewidth=0.8,label="n$_{max}$ = 6")
plt.plot(time,charges3k,marker='d',color='b',ms=2,linewidth=0.8,label="n$_{max}$ = 12")
plt.plot(time,charges4k,marker='d',color='g',ms=2,linewidth=0.8,label="n$_{max}$ = k")

plt.grid()
plt.legend()
plt.show()