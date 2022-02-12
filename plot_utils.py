import numpy
import numpy as np
import scipy
import scipy.spatial
import torch

import matplotlib
from matplotlib import pyplot as plt

def plot_flips(data, flip_order, params, imfile=None):
    fracs = np.linspace(0.,1.,11)

    f,axs = plt.subplots(1,2, figsize=(7.6,4.))

    for flip_case, flip_data  in data.items():


        for k in flip_order:
            v = flip_data[k]

            c, ls, labelstr = params[k]

            if 'lrp' in k:
                lw = 2.1
            else:
                lw = 2


            if flip_case =='generate':
                axs[0].plot(fracs, np.nanmean(v['E'], axis=0), label=labelstr, color=c, linestyle = ls, linewidth=lw)

            if flip_case == 'pruning':
               # axs[1].plot(fracs, np.nanmean(v['E'], axis=0), label=labelstr, color=c, linestyle = ls, linewidth=lw)
                axs[1].plot(fracs, np.nanmean(v['M'], axis=0), label=labelstr, color=c, linestyle = ls, linewidth=lw)


            print(len(v['M']))



        axs[0].set_xlabel('% of nodes added', fontsize=15, y =0.9)
#        axs[0].set_ylabel('AUAC', fontsize=15)
        axs[0].set_ylabel(r'$p_c(x)$', fontsize=20)
        custom_ticks = [0.50, 0.6,0.7,0.8,0.9,1.0]# #0.75,1.00]#,  0.6]
        axs[0].set_yticks(custom_ticks)
        axs[0].set_yticklabels(custom_ticks, fontsize=16)




   #     axs[1].set_xlabel('% of nodes removed', fontsize=15 ,y =0.9)
   #     axs[1].set_ylabel('AUPC', fontsize=15)
   #     custom_ticks = [0.50, 0.6,0.7,0.8]
   #     axs[1].set_yticks(custom_ticks)
   #     axs[1].set_yticklabels(custom_ticks, fontsize=16)



        axs[1].set_xlabel('% of nodes removed', fontsize=15, y =0.9)
     #   axs[1].set_ylabel('MSE', fontsize=15)
        axs[1].set_ylabel(r'$(y_0-y_{m_t})^2$', fontsize=20)
        custom_ticks = range(10)[::2]
        axs[1].set_yticks(custom_ticks)
        axs[1].set_yticklabels(custom_ticks, fontsize=16)


        for ax in axs.flatten():
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)


            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.tick_params(axis='both', which='minor', labelsize=14)

       # plt.legend()


    axs[0].set_title('Activation', fontsize=16, y=1.05)
    axs[1].set_title('Pruning', fontsize=16,  y=1.05) #, x=1.,


   # h1, l1 = axs[0].get_legend_handles_labels()
    
   # lgd = axs[1].legend(h1, l1, loc=8, ncol=len(flip_order), bbox_to_anchor=(-0.73,-0.40, 0, 0), fontsize=12)       
   # lgd = axs[1].legend(h1, l1,  bbox_to_anchor=(0.1,-0.4, 0,0), loc=9, ncol=3, fontsize=12)
    
    
    #ncol=5, fontsize=12 , bbox_to_anchor=(0.3,-.4)) #, 2.2,-0.15)) #(0.0,-0.55, 0, 0))
    
    # ax.grid('on')
    text = axs[1].text(-0.1,1.05, "", transform=ax.transAxes)

    plt.subplots_adjust(wspace=0.38, hspace=0.5)
    f.tight_layout()
    
    
    h1, l1 = axs[0].get_legend_handles_labels()
    h2, l2 = axs[1].get_legend_handles_labels()

    #Shrink the subplots to make room for the legend
    box = axs[0].get_position()
    axs[0].set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.95])
    box = axs[1].get_position()
    axs[1].set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.95])
    #Make the legend
    
    axs[0].legend(h1, l1,  bbox_to_anchor=(0,-.07, 2.2,-0.15), loc=9,
               ncol=3, fontsize=13)

    
    
    
    if imfile is not None:
        f.savefig(imfile, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.show()
    
    
    

def plot_conservation(conservation_dict, savefile):

    f, ax = plt.subplots(1,1, figsize=(6, 4))
    params = {
        'gi': ('#a1a1a1',None,'GI'),
        'gi_detach_KQ_LNorm_Norm': ('black','black',r'LRP (AH+LN)'),   }




    for case in ['gi', 'gi_detach_KQ_LNorm_Norm']:
        v = conservation_dict[case]

        rs = [np.sum(x[0]) for x in v]

        fs = [x[1] for x in v]




        plt.scatter(fs, rs, label=params[case][2], color=params[case][0], s=16) #, fill)

    plt.plot(fs,fs, color='black', linestyle='-', linewidth=1)

    ax.set_ylabel('$\sum_i R_i$', fontsize=25)
    ax.set_xlabel('output $f$', fontsize=25)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.legend(fontsize=15,  markerscale=2)

    f.tight_layout()
    f.savefig(savefile, dpi=300)
    plt.show()
    #rgi, res['gi']