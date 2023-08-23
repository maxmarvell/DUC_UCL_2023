import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os 
import re

'''
    Plot data and store in graph repository
'''

def main():
    q = 2
    plot_light_cone('truncated_path_k3',q)

def plot_light_cone(path:str, q:int):

    rstr = r'heatmap_' + str(q) + r'_([0-9]*)_([0-9e\-]*).csv'
    rx = re.compile(rstr)

    for _, _, files in os.walk('data/'+path):
        for file in files:

            res = rx.match(file)

            if not res:
                continue

            e,seed = res.group(2),res.group(1)

            df = pd.read_csv('data/'+path+'/'+file,index_col=0)
            df = df.reindex(sorted(df.columns,key=lambda num: float(num)), axis=1)
            df = df.fillna(0)
            # df = df.abs()

            plt.figure(1)

            plt.imshow(df.iloc[::-1],cmap='hot',interpolation='nearest')

            try:
                plt.savefig(f'graphs/heatmap/'+path+f'/{q}_{seed}_{e}.png')
            except:
                os.mkdir('graphs/heatmap/'+path)
                plt.savefig(f'graphs/heatmap/'+path+f'/{q}_{seed}_{e}.png')

            plt.clf() 

            plt.figure(2)

            LC = [df.loc[label, str(label)] for label in df.index if str(label) in df.columns]

            plt.plot(range(len(LC)),LC)

        try:
            plt.savefig(f'graphs/light_cone_decay/'+path+f'/{q}_{seed}.png')
        except:
            os.mkdir('graphs/light_cone_decay/'+path)
            plt.savefig(f'graphs/light_cone_decay/'+path+f'/{q}_{seed}.png')

if __name__ == '__main__':
    main()