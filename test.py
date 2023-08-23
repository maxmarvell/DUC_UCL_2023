import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator
import re
import os
import math

# path = 'data/RandomTensorExact/2_1e-07_0.1.csv'
# df = pd.read_csv(path,delimiter=',',index_col=0)
# # cross_section = np.abs(canvas)
# # plt.imshow(cross_section,cmap='hot',interpolation='nearest')
# # df = df.fillna(0)
# plt.imshow(df.iloc[::-1],cmap='hot',interpolation='nearest')

# plt.show()

A = np.array([[1,2,3],
              [2,5,7],
              [5,8,7],
              [9,0,9]])

print(A.shape)

U, S, V = np.linalg.svd(A)

print(U.shape,S.shape,V.shape)

# P = np.loadtxt(f'data/FoldedPertubations/P_2_3714735801_7e-07.csv',
#                                     delimiter=',',dtype='complex_')


# os.mkdir('hello/hello/hello/')
# for T in range(2*5):

#     t = float(T)/2

#     for x in range(-T,T+1):

#         x = float(x)/2

#         print("location: ",x)
#         print("time: ",t)

#         print("x_h: ",math.ceil(t-x),"x_v: ",math.floor(t+1+x))

#         print('\n')

# rstr = r'DU_' + str(q) + r'_([0-9]*).csv'
# rx = re.compile(rstr)

# for _, _, files in os.walk("data/FoldedTensors"):
#         for file in files[:]:
#             res = rx.match(file)
#             seed_value = res.group(1)

#             for e in np.linspace(0.01,0.1,11):
                  
#                 e = str(np.round(e,4)).ljust(5,'0')

#                 f, ax = plt.subplots(2)

#                 df = pd.read_csv(f'data/TnMethod/heatmap_{q}_' + e + f'_{seed_value}.csv')
#                 sns.heatmap(df.iloc[::-1],ax=ax[0],norm=LogNorm())

#                 df = pd.read_csv(f'data/PiMethod/heatmap_{q}_' + e + f'_{seed_value}.csv')
#                 sns.heatmap(df.iloc[::-1],ax=ax[1],norm=LogNorm())

#                 try:
#                     f.savefig(f'graphs/heatmap/{q}_{seed_value}_' + e + '.jpg')
#                 except:
#                     os.mkdir('graphs/heatmap/')
#                     f.savefig(f'graphs/heatmap/{q}_{seed_value}_' + e + '.jpg')

#                 plt.close()

# q = 2

# for _, _, files in os.walk("data/FoldedPertubations"):
#     for file in files:
#         rstr = r'P_' + str(q) + r'_(0.[0-9]*)_([0-9]*).csv'
#         rx = re.compile(rstr)

#         res = rx.match(file)
#         newfile = np.round(float(res.group(1)),3)
#         newfile = str(newfile).ljust(5,'0')

#         path = "data/FoldedPertubations/"
#         oldfile = 'P_' + str(q) + f'_{res.group(1)}_' + res.group(2) + '.csv'
#         newfile = 'P_' + str(q) + f'_{newfile}_' + res.group(2) + '.csv'

#         os.rename(path+oldfile,path+newfile)


# rstr = r'DU_' + str(q) + r'_([0-9]*).csv'
# rx = re.compile(rstr)

# for _, _, files in os.walk("data/FoldedTensors"):
#     for file in files[:1]:
#         res = rx.match(file)
#         seed_value = res.group(1)

#         rstr2 = 'P_' + str(q) + r'_(0.[0-9]*)_' + int(seed_value) + '.csv'
#         rx2 = re.compile(rstr2)

#         rx2.match()


