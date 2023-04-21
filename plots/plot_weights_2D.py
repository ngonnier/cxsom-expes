import pycxsom as cx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.image as mpimg
import os
import sys
from correlation_ratio import *
MAP_SIZE = 100
from PIL import Image
import colorsys
import math
from colormap import *



plt.rc('text', usetex=True)
plt.rcParams['font.size'] = '16'


def find_best_rowcol(n):
    min_r = n
    a = 0
    m = int(np.sqrt(n))
    for i in range(1,m+1):
        r = n % i
        q = n/i
        if(r<min_r):
            min_r = r
            a = i
    rows = i
    if r == 0:
        cols = int(n/i)
    else:
        cols = int(n/i) + 1
    return cols,rows

def plot_gridview_without_random(we,ax,color,step=1,a=0.7):
    def plot_without_random(line):
        difference = np.array([np.sqrt((line[i,0]-line[i-1,0])**2 + (line[i,1]-line[i-1,1])**2)*np.sqrt((line[i,0]-line[i+1,0])**2 + (line[i,1]-line[i+1,1])**2) for i in range(1,len(line)-1)])
        mask = difference < 1
        mask = np.insert(mask,0,True)
        mask = np.append(mask,True)

        ax.plot([w[0] for w in line[mask]], [w[1] for w in line[mask]],c=color,alpha=a)
    plot_without_random(we[0,:])
    plot_without_random(we[-1,:])
    plot_without_random(we[:,0])
    plot_without_random(we[:,-1])


    #plot the first and last rows
    # ax.plot([w[0] for w in we[0,:]], [w[1] for w in we[0,:]],c=color,alpha=a)
    # ax.plot([w[0] for w in we[-1,:]], [w[1] for w in we[-1,:]],c=color,alpha=a)
    # ax.plot([w[0] for w in we[:,0]], [w[1] for w in we[:,0]],c=color,alpha=a)
    # ax.plot([w[0] for w in we[:,-1]], [w[1] for w in we[:,-1]],c=color,alpha=a)
    #plot the rest
    for i in range(1,we.shape[0],step):
        #     ax.plot([w[0] for w in we[i,:]], [w[1] for w in we[i,:]],c=color,alpha=a)
        plot_without_random(we[i,:])

    for j in range(1,we.shape[1],step):
        #     ax.plot([w[0] for w in we[:,j]], [w[1] for w in we[:,j]],c=color,alpha=a)
        plot_without_random(we[i,:])

def plot_gridview(we,ax,color,step=1,a=0.7):
    #mark the 0,0 corner with a point_test
    print(we[0,0])

    ax.scatter([we[0,0][0]], [we[0,0][1]],c = 'r',s=50,edgecolor='k')
    ax.arrow(we[0,0][0],  we[0,0][1], we[step,0][0] - we[0,0][0], we[step,0][1] - we[0,0][1] , color='r',width=0.005)
    ax.arrow(we[0,0][0],  we[0,0][1], we[0,step][0] - we[0,0][0], we[0,step][1] - we[0,0][1] , color='r',width=0.005)
    #plot the first and last rows
    ax.plot([w[0] for w in we[0,:]], [w[1] for w in we[0,:]],c=color,alpha=a)
    ax.plot([w[0] for w in we[-1,:]], [w[1] for w in we[-1,:]],c=color,alpha=a)
    ax.plot([w[0] for w in we[:,0]], [w[1] for w in we[:,0]],c=color,alpha=a)
    ax.plot([w[0] for w in we[:,-1]], [w[1] for w in we[:,-1]],c=color,alpha=a)
    #plot the rest
    for i in range(step,we.shape[0],step):
        ax.plot([w[0] for w in we[i,:]], [w[1] for w in we[i,:]],c=color,alpha=a)
    for j in range(step,we.shape[1],step):
        ax.plot([w[0] for w in we[:,j]], [w[1] for w in we[:,j]],c=color,alpha=a)





if __name__ == "__main__":
    if len(sys.argv)<3:
        sys.exit("usage: python3 plot_weights.py <test_number> <closed>  <opt> <time_inp> <data dir> <begin> <end> <step> <inputs> ")
    tnum = int(sys.argv[1])
    closed = int(sys.argv[2])
    opt = sys.argv[3]
    time_inp = int(sys.argv[4])
    directories = sys.argv[5]
    begin = int(sys.argv[6])
    end = int(sys.argv[7])
    step = int(sys.argv[8])
    inputs = sys.argv[9:]

    path = os.path.join(directories, 'out')
    maps = os.listdir(path)
    inp_path = os.path.join(directories,'ztest-in')
    cmap = colorwheel()
    cmap_f  = np.vectorize(colorwheel_f)
        #inputs = os.listdir(inp_path)
    analysis_prefix = "zfrz"
    if closed >0:
        analysis_prefix = f"zclosed-{closed}"

    def varpath(name,timeline):
        return os.path.join(timeline, name)

    #open inputs
    input_dict=dict()
    for inp in inputs:
        with cx.variable.Realize(os.path.join(inp_path,inp+".var")) as input:
            r = input.time_range()
            input_dict[inp] = np.array([input[at] for at in range(r[0],r[1])])
    #open maps

    #CORRECTION BUG 2SOM_S_005
    uv = input_dict['U']
    # uv[0:10999,0] = 2.0 * uv[:10999,0]
    input1 = input_dict['I1'][:,0]
    #print(input1)
    #choose inputs with same value for X(1) but not X(2)
    positions = np.argwhere((input1 > 0.8) & (input1 < 0.82))
    #print(input_dict['I1'][positions])
    index1 = positions[0][0]
    index2 = positions[1][0]
    #
    uv1 = uv[index1]
    uv2 = uv[index2]

    #print(uv1)
    #print(uv2)

    # I1_1 = input_dict['I1'][index1]
    # I1_2 = input_dict['I1'][index2]
    # #
    # I2_1 = input_dict['I2'][index1]
    # I2_2 = input_dict['I2'][index2]

    def plot_image(timestep):
        c,r = find_best_rowcol(len(maps))
        #poids externes
        fig,ax = plt.subplots(r,c+1,squeeze = False,layout='tight')
        fig.suptitle("Poids Externes")
        fig_grid,ax_grid = plt.subplots(r,c,figsize=(20,10),layout='tight')
        ax_grid=np.reshape(ax_grid,(r,c))

        fig6,ax6 = plt.subplots(r,c+1,squeeze = False)
        fig6.suptitle("Poids Contextuels")
        fig3,ax3 = plt.subplots(r,c+1)
        ax=np.reshape(ax,(r,c+1))
        ax3 = np.reshape(ax3,(r,c+1))
        #inputs
        fig5,ax5 = plt.subplots(r,c+1,squeeze = False)
        colormap2D = np.array([[[i/MAP_SIZE, j/MAP_SIZE] for j in range(MAP_SIZE)] for i in range(MAP_SIZE)])
        cmap = colormap_a(colormap2D,'ziegler')
        ax[-1,-1].imshow(cmap,extent=(0,1,0,1),origin='lower')

        ax[-1,-1].set_title('colormap')
        ax[-1,-1].set_xlabel('première coordonnée')
        ax[-1,-1].set_ylabel('deuxième coordonnée')

        ax5[-1,-1].imshow(cmap,extent=(0,1,0,1),origin='lower')
        ax5[-1,-1].set_title('colormap')
        ax5[-1,-1].set_xlabel('première coordonnée')
        ax5[-1,-1].set_ylabel('deuxième coordonnée')
        ax3[-1,-1].imshow(cmap,extent=(0,1,0,1),origin='lower')
        ax3[-1,-1].set_title('colormap')
        ax3[-1,-1].set_xlabel('$U_0$')
        ax3[-1,-1].set_ylabel('$U_1$')
        #ax3[-1,-1].scatter([uv1[0]], [uv1[1]], c = ['c'], s=50, edgecolors = 'k',label = f"$X^{(1)} = ({input_dict['I1'][index1,0]:.02f}, {input_dict['I1'][index1,1]:.02f})$, $X^{(2)} = ({input_dict['I2'][index1,0]:.02f}, {input_dict['I2'][index1,1]:.02f})$,$U = ({uv[index1,0]:.02f}, {uv[index1,1]:.02f})$")
        #ax3[-1,-1].scatter([uv2[0]], [uv2[1]], c = ['r'], s=50, edgecolors = 'k', label = f"$X^{(1)} = ({input_dict['I1'][index2,0]:.02f}, {input_dict['I1'][index2,1]:.02f})$, $X^{(2)} = ({input_dict['I2'][index2,0]:.02f}, {input_dict['I2'][index2,1]:.02f})$,$U = ({uv[index2,0]:.02f}, {uv[index2,1]:.02f})$")

        if(len(maps ) ==3):
            print("coucou")
            fig4,ax4 = plt.subplots(2*r,c+1,squeeze = False,figsize=(30,8))
            ax4[r+row,col].imshow(colormap_a(weight_dict['Wc-1.var'],'ziegler'),extent=(0,1,0,1))
            ax4[r-1,-1].imshow(cmap,extent=(0,1,0,1),origin='lower')
        else:
            fig4,ax4 = plt.subplots(r,c+1,squeeze = False,figsize=(16,5))

        for i,m in enumerate(maps):
            col_inp = iter(['green','magenta','black','grey'])
            row,col = int(i/c), i%c
            try:
                with cx.variable.Realize(os.path.join(directories,analysis_prefix+f'-%04d-out'%timestep,m,'BMU.var')) as bmu:
                    tr = bmu.time_range()
                    bmus = np.array([bmu[at] for at in range(tr[0],tr[1])])
            except:
                pass

            if m == 'M1':
                key = 'I1'
                color = 'k'
            elif m == 'M2':
                key = 'I2'
                color = 'w'
            else:
                key = 'I3'

            weight_dict = dict()
            weights = os.listdir(os.path.join(directories,'wgt',m))
            for w in weights:
                wf= cx.variable.Realize(varpath(w,os.path.join(directories,analysis_prefix+f'-%04d-wgt'%timestep,m)))
                wf.open()
                weight_dict[w] = wf[0]
                wf.close()


            ax[row,col].imshow(colormap_a(weight_dict['We-0.var'],'ziegler'),extent=(0,1,0,1))
            ax4[row,col].imshow(colormap_a(weight_dict['Wc-0.var'],'ziegler'),extent=(0,1,0,1))


            ax4[-1,-1].imshow(cmap,extent=(0,1,0,1),origin='lower')
            ax4[-1,-1].set_title('colormap')
            ax4[-1,-1].set_xlabel('$\\omega_c$, coordonnée $0$ : $p|_x$')
            ax4[-1,-1].set_ylabel('$\\omega_c$, coordonnée $1$ : $p|_y$')
            #print(r)
            #print(row)

            #ax4[-1,-1].scatter([weight_dict['Wc-0.var'][i,j,0] for i in range(0,weight_dict['Wc-0.var'].shape[0]) for j in range(0,weight_dict['Wc-0.var'].shape[0],10)], [weight_dict['Wc-0.var'][i,j,1] for i in range(weight_dict['Wc-0.var'].shape[0]) for j in range(weight_dict['Wc-0.var'].shape[0])])
            plot_gridview(weight_dict['Wc-0.var'], ax4[-1,-1], color, 10 ,a=0.2)
            #ax4[row,col].scatter([bmus[index1,0], bmus[index2,0]],[bmus[index1,1], bmus[index2,1]] ,s = 50 , c=['c', 'r'], edgecolors='black')
            plot_gridview(weight_dict['We-0.var'],ax_grid[row,col], 'k',step=10)
            ax_grid[row,col].set_title(f'$\omega_e$, carte $M^{i+1}$')



            #for key in input_dict.keys():
            #    print(key)


             #cr,phi= correlation_ratio_ND(input_dict['U'],bmus,100)
            #cr_test,_ = correlation_ratio_ND(input_dict['U'],input_dict['U'],100)
            #print(cr_test)
            #u = np.reshape(input_dict['U'], (len(input_dict['U']),1))
            #v = np.reshape(input_dict['V'], (len(input_dict['V']),1))
            #uv = np.concatenate([np.reshape(input_dict['U'], (len(input_dict['U']),1)), np.reshape(input_dict['V'], (len(input_dict['V']),1))], axis = -1)

            print(uv.shape)
            #ax6.scatter(uv[:,0], uv[:,1])
            #ax6.scatter([uv1[0], uv2[0]], [uv1[1], uv2[1]], c=['c', 'r'], s=50, edgecolors = 'k') #point 1 : cyan, point 2 : red
            #im,eu = mi_conj_n(bmus,uv,50,100)
            # cr,phi = correlation_ratio_2D(uv,bmus,100)
            # cr_i1,phi = correlation_ratio_2D(uv,input_dict['I1'],100)
            # cr_i2,phi = correlation_ratio_2D(uv,input_dict['I2'],100)
            # print(f"eta(U,I1) = {cr_i1}")
            # print(f"eta(U,I2) = {cr_i2}")

            #colors = np.concatenate([uv, np.zeros((input_dict[key].shape[0],1))],axis=-1)
            #sc = ax3[row,col].scatter(bmus[:,0],bmus[:,1],s = 10, c=colormap_l(uv,type='ziegler',n=len(bmus[:,0])))
            #ax3[row,col].scatter([bmus[index1,0], bmus[index2,0]],[bmus[index1,1], bmus[index2,1]] ,s = 50 , c=['c', 'r'], edgecolors='black')


            if(m=='M1'):
                ax4[row,col].set_title("$\\omega_c^{(1)}$, carte $M^{(1)}$")
                ax3[row,col].set_title("Carte $M^{(1)}$")
                ax3[row,col].set_xlabel('$\\Pi^{(1)}|_x$')
                ax3[row,col].set_ylabel('$\\Pi^{(1)}|y$')
                ax4[row,col].set_xlabel('$p|_x$')
                ax4[row,col].set_ylabel('$p|_y$')
            elif(m=='M2'):
                ax4[row,col].set_title("$\\omega_c^{(2)}$, carte $M^{(2)}$")
                ax3[row,col].set_title("Carte $M^{(2)}$")
                ax3[row,col].set_xlabel('$\\Pi^{(2)}|_x$')
                ax3[row,col].set_ylabel('$\\Pi^{(2)}|_y$')
                ax4[row,col].set_xlabel('$p|_x$')
                ax4[row,col].set_ylabel('$p|_y$')
            else:
                ax4[row,col].set_title("$\\omega_c^{(3)}$, carte $M^{(3)}$")
                ax3[row,col].set_title("Carte $M^{(3)}$")
                ax3[row,col].set_xlabel('$\\Pi^{(3)}|_x$')
                ax3[row,col].set_ylabel('$\\Pi^{(3)}|_y$')
                ax4[row,col].set_xlabel('$p|_x$')
                ax4[row,col].set_ylabel('$p|_y$')

            #ax3[row,col].text(0 , 1.2, f'$\\eta(U|BMU) = {cr:.3f}$')
            fig3.tight_layout()
            #fig3.suptitle('$U$ fonction de $\\Pi$ ')


            ax[row,col].set_xlabel('Map Positions')
            ax[row,col].set_ylabel('Map Positions')
            #ax5[row,col].scatter(bmus[:,0],bmus[:,1],c=colormap_l(input_dict[key],type='ziegler',n=len(bmus[:,0])),label=f"{key[:-4]}")
            ax5[row,col].axis('square')
            ax5[row,col].set_title(f'Entrées $X^{i+1}$, carte $M^{i+1}$')
            #ax[row,col].scatter(bmus[MAP_SIZE],input_dict[key][MAP_SIZE],alpha=1,s=70,color = 'k', label=f"{key[:-4]}")

            ax[row,col].set_title(f'Carte {m}')

        handles, labels = ax3[-1,-1].get_legend_handles_labels()
        fig3.legend(handles, labels, loc='lower center', ncol=2)
        ax[row,col].legend()
        fig.tight_layout()
        fig4.tight_layout()
        fig3.tight_layout()
        #fig.savefig(f'{directories}weights.svg')
        return fig,fig4, fig5, fig_grid

    if opt=='image':
        fig,fig4, fig5, fig_grid =  plot_image(tnum)
        fig_grid.savefig(f'{directories}/weights_externe-%06d.png'%tnum)
        fig4.savefig(f'{directories}/weights_contexte-0-%06d.png'%tnum)
        #colorwheel()
        plt.show()

    if opt =='film':
        idx = 0
        for tnum in range(begin,end,step):
            fig,fig4, fig5, fig_grid = plot_image(tnum)
            fig.savefig(f'{directories}//weights_externe-%06d.png'%idx)
            fig4.savefig(f'{directories}//weights_contexte-0-%06d.png'%idx)
            #fig6.savefig(f'{directories}/weights_contexte-1-%06d.png'%idx)
            fig_grid.savefig(f'{directories}/weights_grid-%06d.png'%idx)
            #fig_grid_c.savefig(('../2som2D_in_anneau4D_figures/weights_grid_contexte-%06d.png'%idx))
            plt.close('all')
            idx = idx+1
    #plt.show()
