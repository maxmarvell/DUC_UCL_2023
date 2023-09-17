import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


path = "./Runtime_Scores.csv"
data = pd.read_csv(path)
runtimes1k = data["1,k"]
runtimes2k = data["2,k"]
runtimes3k = data["3,k"]
runtimes4k = data["4,k"]
time = range(48)
threshold = np.full(48,300.0)


fig, ax = plt.subplots()
plt.rcParams["font.family"] = "Times New Roman"
plt.xlabel("t", size=14, fontname="Times New Roman", labelpad=10)
plt.ylabel("Runtime (s)" ,size=14, fontname="Times New Roman", labelpad=10)
plt.title("Runtime to Compute the Lightcone up to t$^{th}$ Layer", fontweight ='bold',size=14, fontname="Times New Roman")
ax.minorticks_on()
major_ticks = np.arange(time[0],time[-1],10)
minor_ticks = np.arange(time[0],time[-1],0.5)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.grid(which='minor', linewidth=0.5, alpha=0.5)

plt.plot(time,runtimes1k,marker='d',color='k',ms=2,linewidth=0.8,label="d = 1")
plt.plot(time,runtimes2k,marker='d',color='r',ms=2,linewidth=0.8,label="d = 2")
plt.plot(time,runtimes3k,marker='d',color='b',ms=2,linewidth=0.8,label="d = 3")
plt.plot(time,runtimes4k,marker='d',color='g',ms=2,linewidth=0.8,label="d = 4")
plt.plot(time,threshold,color='k',ms=2,linewidth=1,linestyle="dashed",label="5 minute mark")

plt.grid()
plt.legend()
plt.show()