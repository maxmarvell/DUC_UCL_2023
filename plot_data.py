from plotly.subplots import make_subplots
from matplotlib import cm
from cross_sections import fit
import plotly.graph_objects as go
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
    path = 'PathIntegralTree'
    graph_3(path,[5.5,11.5])

def graph_1(path:str):

    '''
        plot graph of heatmap using path integral and
        exact method, heatmap of error and conserved charges
    '''

    rstr = r'([0-9])_([0-9]*)_([0-9e\-]*).csv'
    rx = re.compile(rstr)

    for _, _, files in os.walk(f'data/{path}'):
        for file in files:

            res = rx.match(file)

            if not res:
                continue

            exact = np.loadtxt(f'data/TensorExact/{res.group(0)}',
                               dtype='complex_',
                               delimiter=',')
            
            approx = np.loadtxt(f'data/{path}/{res.group(0)}',
                                dtype='complex_',
                                delimiter=',')
            
            diff = approx.shape[0] - exact.shape[0]
            RE = exact-approx[diff::,diff:-diff]
            RE2 = exact/np.linalg.norm(exact)-approx[diff::,diff:-diff]/np.linalg.norm(approx[diff::,diff:-diff])

            plt.rcParams['font.family'] = 'Times New Roman'

            f = plt.figure(figsize=(5.2,7.4))
            subf1, subf2 = f.subfigures(2,1,height_ratios=[9,2],hspace=0)

            axes = subf1.subplots(3,1,sharex=True)
            subf1.subplots_adjust(hspace=0.01,top=0.9,bottom=0.1,left=0.1)
            cbar_ax = subf1.add_axes([.91, .3, .03, .4])

            axes[0].annotate(f'local Hilbert space dimension: {res.group(1)}',
                             xy=(-0.1,1.25),xycoords='axes fraction',
                             fontweight='bold')
            axes[0].annotate(f'Pertubation strength e: {res.group(3)}',
                             xy=(-0.1,1.15),xycoords='axes fraction',
                             fontweight='bold')
            axes[2].annotate(f'Norm: {np.round(np.linalg.norm(RE2),3)}',
                             xy=(0.01,0.92),xycoords='axes fraction',
                             color='white',fontweight='bold')
            axes[0].annotate(r'$\left\langle Z_x|Z(t)\right\rangle$',
                             xy=(1.,0.33),xycoords='axes fraction',size=9)
            
            axes[0].tick_params(bottom=False)
            axes[1].tick_params(bottom=False)

            df_exact = pd.DataFrame(np.abs(exact),
                                    columns=[i/2 for i in range(-exact.shape[0]+2,exact.shape[0])],
                                    index=[i/2 for i in range(exact.shape[0]-1,-1,-1)])
            df_approx = pd.DataFrame(np.abs(approx[diff::,diff:-diff]),
                                    columns=[i/2 for i in range(-exact.shape[0]+2,exact.shape[0])],
                                    index=[i/2 for i in range(exact.shape[0]-1,-1,-1)]) 
            df_RE = pd.DataFrame(np.abs(RE),
                                    columns=[i/2 for i in range(-RE.shape[0]+2,RE.shape[0])],
                                    index=[i/2 for i in range(RE.shape[0]-1,-1,-1)]) 
            
            for i, (ax, df, label) in enumerate(zip(axes,
                                                    [df_exact,df_approx,df_RE],
                                                    ['Exact','Approximation','Difference'])):
                
                sns.heatmap(df, ax = ax,
                            cbar= i == 0,
                            vmin = 0, vmax = 1,
                            cmap = 'hot',
                            cbar_ax = None if i else cbar_ax
                )
                
                ax.annotate(label,xy=(0.01,0.05),xycoords='axes fraction',
                            color='white',fontweight='bold')
                
            axes[-1].set_xlabel('x', size=14, fontname='Times New Roman')
            axes[1].set_ylabel('t', size=14, fontname='Times New Roman')

            err_exact  = np.loadtxt(f'data/TensorExact/charge_conservation/QC_{res.group(0)}',
                                    delimiter=',')
            
            err_approx  = np.loadtxt(f'data/{path}/charge_conservation/QC_{res.group(0)}',
                                     delimiter=',')
            
            tspan = [i/2 for i in range(exact.shape[0])]

            subf2.subplots_adjust(bottom=.35,top=.96)
            ax = subf2.subplots()
            ax.plot(tspan,err_exact,tspan,err_approx[:-diff])
            ax.set_ylim([0,1.2])
            ax.set_xlabel('t', size=14, fontname='Times New Roman')
            ax.set_ylabel(r'$SUM\left(\left\langle Z_x|Z(t)\right\rangle\right)$', size=9.5, fontname='Times New Roman')

            try:
                f.savefig(f'graphs/graph_1/{path}_{res.group(1)}_{res.group(2)}_{res.group(3)}.png')
            except:
                os.mkdir('graphs/graph_1')
                f.savefig(f'graphs/graph_1/{path}_{res.group(1)}_{res.group(2)}_{res.group(3)}.png')

def graph_2(path:str):
    
    '''
        Plot graph showing 3d correlation function plots
        where color corresponds to phase of correlation at
        that point
    '''

    rstr = r'([0-9])_([0-9]*)_([0-9e\-]*).csv'
    rx = re.compile(rstr)

    for _, _, files in os.walk(f'data/{path}'):
        for file in files:

            res = rx.match(file)

            if not res:
                continue

            data = np.loadtxt(f'data/{path}/{res.group(0)}',
                              dtype='complex_',
                              delimiter=',')

            x_inphase,x_outphase = [],[]
            t_inphase,t_outphase = [],[]
            data_inphase,data_outphase = [],[]

            for i,t in enumerate(range(data.shape[0]-1,-1,-1)):
                for j,x in enumerate(range(-data.shape[0]+2,data.shape[0])):
                    if int(2*(t/2+x/2))%2 == 0:
                        x_inphase.append(x/2)
                        t_inphase.append(t/2)
                        data_inphase.append(data[i,j])
                    else:
                        x_outphase.append(x/2)
                        t_outphase.append(t/2)
                        data_outphase.append(data[i,j])

            plt.rcParams['font.family'] = 'Times New Roman'

            f = make_subplots(
                rows = 1,
                cols = 2,
                specs = [[{'type':'scene'},{'type':'scene'}]],
                subplot_titles = ('In-phase Sites', 'Out-phase sites') 
            )

            MeshInphase = go.Mesh3d(
                x = x_inphase,
                y = t_inphase,
                z = np.abs(data_inphase),
                intensity = np.angle(data_inphase),
                colorscale = 'viridis',
                cmax = np.pi,
                cmin = -np.pi,
            )

            MeshOutphase = go.Mesh3d(
                x = x_outphase,
                y = t_outphase,
                z = np.abs(data_outphase),
                intensity = np.angle(data_outphase),
                colorscale = 'viridis',
                cmax=np.pi,
                cmin=-np.pi,
                showlegend=False,
            )

            f.add_trace(MeshInphase,row=1,col=1)
            f.add_trace(MeshOutphase,row=1,col=2)

            z_max = max(max(MeshInphase.z), max(MeshOutphase.z))
            z_min = -.05

            f.update_scenes(zaxis_range=[z_min, z_max], row=1, col=1)
            f.update_scenes(zaxis_range=[z_min, z_max], row=1, col=2)

            f.update_layout(
                font_family = 'Times New Roman',
                height = 800,
                width = 1400,
                margin = {'b':100,'t':100},
            )

            f.layout.scene1.camera.eye = {'x':1.4,'y':1.4,'z':1.4}
            f.layout.scene1.zaxis.title = ''
            f.layout.scene2.camera.eye = {'x':1.4,'y':1.4,'z':1.4}
            f.layout.scene2.zaxis.title = ''

            f.add_annotation(
                text = r'$\left \langle Z_{x}|Z(t)\right\rangle$',
                x = 0,
                y = 0.5,
                showarrow = False,
                textangle = -90,
            )

            f.add_annotation(
                text = r'$\left \langle Z_{x}|Z(t)\right\rangle$',
                x = 0.55,
                y = 0.5,
                showarrow = False,
                textangle = -90,
            )

            try:
                f.write_image(f'graphs/graph_2/{path}_{res.group(1)}_{res.group(2)}_{res.group(3)}.png')
            except:
                os.mkdir('graphs/graph_2')
                f.write_image(f'graphs/graph_2/{path}_{res.group(1)}_{res.group(2)}_{res.group(3)}.png')

def graph_3(path:str,
            timeslices:np.ndarray):
    
    '''
        Plot graph showing the cross-section corresponding
        to a timeslice on the heatmap
    '''
        
    rstr = r'([0-9])_([0-9]*)_([0-9e\-]*).csv'
    rx = re.compile(rstr)

    for _, _, files in os.walk(f'data/{path}'):
        for file in files[:1]:

            res = rx.match(file)

            if not res:
                continue

            data = np.loadtxt(f'data/{path}/{res.group(0)}',
                              dtype='complex_',
                              delimiter=',')
            
            if any(np.array(timeslices)>data.shape[0]):
                print('Timeslice to far in light cone, not enough data')
                return
            
            df = pd.DataFrame(np.abs(data),
                              columns=[i/2 for i in range(-data.shape[0]+2,data.shape[0])],
                              index=[i/2 for i in range(data.shape[0]-1,-1,-1)])
            
            plt.rcParams['font.family'] = 'Times New Roman'
            
            f, (ax1, ax2) = plt.subplots(1,2,figsize=(14.8,6.2),width_ratios=(5,2))

            sns.heatmap(df, ax = ax1,
                        cbar = True,
                        vmin = 0, vmax = 1,
                        cmap = 'hot',
            )
            
            colours = ['b','c','g','r', 'm','k']
            
            for i,t in enumerate(timeslices):

                ax1.hlines(2*(max(df.columns)-t),*ax1.get_xlim(),color=colours[i])

                cross_section = list((df.loc[t,:].values)**2)
                space = list((np.array(range(len(cross_section)))  + 1 - (len(cross_section)/2))/2)

                if t % 1 == 0:
                    c = 0
                else:
                    c = 1

                space_even = space[c::2]
                space_odd = space[1-c::2]
                cross_section_even = cross_section[c::2]
                cross_section_odd = cross_section[1-c::2]

                ax2.scatter(space_odd, cross_section_odd, marker="d", color='k', s=5)

                try:
                    dense_inputs_even, fit_data_even, R2_even = fit(space_even, cross_section_even)
                    dense_inputs_odd, fit_data_odd, R2_odd = fit(space_odd, cross_section_odd)
                    ax2.plot(dense_inputs_odd, fit_data_odd, color=colours[i], linewidth=1.2, label=f"t = {t}")
                    ax2.plot(dense_inputs_even, fit_data_even, color=colours[i], linewidth=0.5, linestyle='dashed')
                except:
                    pass

            ax1.annotate(f'Local Hilbert space dimension: {res.group(1)}',
                         xy=(-0.1,1.08),xycoords='axes fraction',
                         fontweight='bold')
            ax1.annotate(f'Pertubation strength e: {res.group(3)}',
                         xy=(-0.1,1.04),xycoords='axes fraction',
                         fontweight='bold')
            ax1.annotate(r'$\left\langle Z_x|Z(t)\right\rangle$',
                             xy=(1.05,1.02),xycoords='axes fraction',size=9)

            ax1.tick_params(left=False,bottom=False)
            ax1.set_xlabel('x', size=14)
            ax1.set_ylabel('t', size=14)
            ax2.set_xlabel('x', size=14)
            ax2.set_ylabel(r'$\left|\left\langle Z_x|Z(t)\right\rangle\right|^2$',size=10)
            f.subplots_adjust(top=.9,left=.05,bottom=.12)
            
            ax2.legend()

            try:
                f.savefig(f'graphs/graph_3/{path}_{res.group(1)}_{res.group(2)}_{res.group(3)}.png')
            except:
                os.mkdir('graphs/graph_3')
                f.savefig(f'graphs/graph_3/{path}_{res.group(1)}_{res.group(2)}_{res.group(3)}.png')

if __name__ == '__main__':
    main()