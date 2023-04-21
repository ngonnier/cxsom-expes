import pycxsom as cx
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import cplot

def init_path(dir,pos):
    return os.path.join(dir,'bmus-rlx',pos+'.var')

def out_path(dir,map,timestep,time_inp):
    return os.path.join(dir,f'out-rlx-{timestep}-{time_inp}',map,'BMU.var')

def ag_path(dir,map,timestep,time_inp):
    return os.path.join(dir,f'out-rlx-{timestep}-{time_inp}',map,'Am.var')

def traj_path(dir,map, prefix):
    return os.path.join(dir,prefix+'-rlx',map,'BMU.var')

def frz_path(dir,map,frz_prefix):
    return os.path.join(dir,frz_prefix+'-out',map,'BMU.var')

def xy_to_rgb(tabx,taby):
    return [(x,0,y) for x,y in zip(tabx,taby)]

if __name__ == "__main__":
    if(len(sys.argv)<2):
        sys.exit("Usage: python3 plot_champs.py <path_to_dir> <prefix> <timestep> <time_inp> <ntraj>")
    else:
        dir = sys.argv[1]
        timestep = sys.argv[3]
        time_inp = int(sys.argv[4])
        ntraj = int(sys.argv[5])
        prefix = sys.argv[2]

    MAP_SIZE = 500
    delta = 0.05
    #champ des argmax
    with cx.variable.Realize(init_path(dir,'P1')) as p1:
        with cx.variable.Realize(init_path(dir,'P2')) as p2:
            pos1 = np.array([p1[i] for i in range(p1.time_range()[1]+1)])
            pos2 = np.array([p2[i] for i in range(p2.time_range()[1]+1)])

    with cx.variable.Realize(out_path(dir,'M1',timestep, time_inp)) as b1:
        with cx.variable.Realize(out_path(dir,'M2',timestep, time_inp)) as b2:
            out11 = np.array([b1[i] for i in range(b1.time_range()[1]+1)])
            out21 = np.array([b2[i] for i in range(b2.time_range()[1]+1)])

    with cx.variable.Realize(ag_path(dir,'M1',timestep, time_inp)) as b1:
        with cx.variable.Realize(ag_path(dir,'M2',timestep, time_inp)) as b2:
            ag11 = np.array([b1[i] for i in range(b1.time_range()[1]+1)])
            ag21= np.array([b2[i] for i in range(b1.time_range()[1]+1)])

    #trajectoires

    def fleche(pos,out):
        signe = np.sign(out-pos)
        return signe*min(delta,abs(out-pos))

    def fleche2(pos,out):
        signe = np.sign(out-pos)
        return out-pos

    fleche_func = np.vectorize(fleche)

    positions1 = pos1.reshape((MAP_SIZE,MAP_SIZE))
    positions2 = pos2.reshape((MAP_SIZE,MAP_SIZE))

    out1 = out11.reshape((MAP_SIZE,MAP_SIZE))
    out2 = out21.reshape((MAP_SIZE,MAP_SIZE))

    ag11 = ag11.reshape((MAP_SIZE,MAP_SIZE,MAP_SIZE))
    ag21 = ag11.reshape((MAP_SIZE,MAP_SIZE,MAP_SIZE))

    ag1 = ag11[0,:,:]
    ag2 = ag21[:,0,:]
    dist2D = abs((positions1-out1)) + abs((positions2 - out2))

    fig9,ax9 = plt.subplots(1,1)
    ax9.quiver(positions2[::10,::10],positions1[::10,::10],fleche_func(positions2[::10,::10],out2[::10,::10]),fleche_func(positions1[::10,::10],out1[::10,::10]),headwidth=2)
    #ax9.quiver(positions2,positions1,fleche_func(positions2,out2),fleche_func(positions1,out1),headwidth=2)


    fig1,ax1 = plt.subplots(1,1)
    im1 = ax1.imshow(np.abs(positions1-out1),extent=[0,1,0,1],origin='lower')
    fig1.colorbar(im1,ax=ax1)

    fig2,ax2 = plt.subplots(1,1)
    im2 = ax2.imshow(np.abs(positions2-out2),extent=[0,1,0,1],origin='lower')
    fig2.colorbar(im2,ax=ax2)

    fig3,ax3 = plt.subplots(1,1)
    im3 = ax3.imshow(dist2D,extent=[0,1,0,1],origin='lower')
    fig3.colorbar(im3,ax=ax3)

    fig4,ax4 = plt.subplots(1,1)
    im4 = ax4.imshow(np.abs(out1),extent=[0,1,0,1],origin='lower')
    fig4.colorbar(im4,ax=ax4)
    ax4.set_xlabel('map Y positions')
    ax4.set_ylabel('map X positions')

    fig,ax = plt.subplots(1,1)
    im= ax.imshow(np.abs(out2),extent=[0,1,0,1],origin='lower')
    fig1.colorbar(im,ax=ax)
    ax.set_xlabel('map Y positions')
    ax.set_ylabel('map X positions')

    fig7,ax7 = plt.subplots(1,1)
    ax7.plot(positions1[:,0],out2[:,0])
    ax7.set_ylabel('argmax Am, map Y')
    ax7.set_xlabel('map X positions')
    #ax7.invert_yaxis()

    fig8,ax8 = plt.subplots(1,1)
    ax8.plot(positions2[0,:],out1[0,:])
    ax8.set_xlabel('map Y positions')
    ax8.set_ylabel('argmax Am, map X')

    fig10,ax10 = plt.subplots(1,2)
    im10 = ax10[0].imshow(ag1,extent=[0,1,0,1],origin='lower')
    im10 = ax10[1].imshow(ag2,extent=[0,1,0,1],origin='lower')
    ax10[0].set_xlabel('map Y positions')
    ax10[0].set_ylabel('map X positions')
    ax10[1].set_xlabel('map Y positions')
    ax10[1].set_ylabel('map X positions')
    fig10.colorbar(im10,ax=ax10[1])

    for i in range(1,ntraj+1):
        with cx.variable.Realize(traj_path(dir,'M1',prefix+"-%03d"%i)) as b1:
            with cx.variable.Realize(traj_path(dir,'M2',prefix+"-%03d"%i)) as b2:
                traj1 = np.array([b1[i] for i in range(b1.time_range()[1]+1)])
                traj2 = np.array([b2[i] for i in range(b2.time_range()[1]+1)])
                ax1.plot(traj2,traj1,'-')
                ax1.scatter(traj2[-1],traj1[-1],c='k',s=50)
                ax2.plot(traj2,traj1,'-')
                ax2.scatter(traj2[-1],traj1[-1],c='k',s=50)
                ax3.plot(traj2,traj1,'-')
                ax3.scatter(traj2[-1],traj1[-1],c='k',s=50)
                ax4.plot(traj2,traj1,'-')
                ax4.scatter(traj2[-1],traj1[-1],c='k',s=50)
                ax.plot(traj2,traj1,'-')
                ax.scatter(traj2[-1],traj1[-1],c='k',s=50)
                ax9.plot(traj2,traj1,'-')
                ax9.scatter(traj2[-1],traj1[-1],c='k',s=50)

    try:
        with cx.variable.Realize(frz_path(dir,'M1','zfrz-%04d'% int(timestep))) as fbmu1:
            with cx.variable.Realize(frz_path(dir,'M2','zfrz-%04d'% int(timestep))) as fbmu2:
                ax1.scatter(fbmu2[time_inp],fbmu1[time_inp],s=100,c='r',marker='X')
                ax2.scatter(fbmu2[time_inp],fbmu1[time_inp],s=100,c='r',marker='X')
                ax3.scatter(fbmu2[time_inp],fbmu1[time_inp],s=100,c='r',marker='X')
                ax4.scatter(fbmu2[time_inp],fbmu1[time_inp],s=100,c='r',marker='X')
                ax9.scatter(fbmu2[time_inp],fbmu1[time_inp],s=100,c='r',marker='X')
    except:
        pass

    plt.show()
