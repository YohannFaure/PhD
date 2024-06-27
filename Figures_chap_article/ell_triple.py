## Imports
# Science
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

main_folder = 'E:/manuscrit_these/'
main_folder = '../'


with open(main_folder + "parameters.py", 'r') as f:
        exec(f.read())




### Fig 6 a.

data_fig_5 = np.load(main_folder + "data_for_figures/figure_5.npy",allow_pickle=True).all()



lc_solid_mean                   = data_fig_5["lc_solid_mean"]
ell_per_lc_round_solid          = data_fig_5["ell_per_lc_round_solid"]
lc_list_round                   = data_fig_5["lc_list_round"]
ell_per_lc_round                = data_fig_5["ell_per_lc_round"]
ell_per_lc_round_width          = data_fig_5["ell_per_lc_round_width"]
ell_per_lc_round_width_solid    = data_fig_5["ell_per_lc_round_width_solid"]


mean_lc                         = data_fig_5["mean_lc"]
lc_event                        = data_fig_5["lc_event"]
ell                             = data_fig_5["ell"]
is_solid                        = data_fig_5["is_solid"]






### Fig 6 d.

### load data : set up the loading

loaded_data = np.load(main_folder + "data_for_figures/Figure_6.npy",allow_pickle=True).all()
locals().update(loaded_data)


solids = [14,15,16,17,18,37,38]

# name of the reference file, containing unloaded signal, usually event-001
loc_file_zero = "event-001.npy"

# parameters file
loc_params="parameters.txt"


# control smoothing of the data (rolling average) and starting point
roll_smooth=10
start=0




### Location of the data inside the file
# channels containing the actual strain gages
gages_channels = np.concatenate([np.arange(0,15),np.arange(16,31)])

# channels containing the normal and tangetial force
forces_channels = [32,33]

# channel containing the trigger
trigger_channel = 34



### Load data : load and create usefull variables
## Parameters
# default x, just in case there's nothing in the saved params
x=np.array([1,2,3,4,5,8,11,12,13,14])*1e-2

import os

chosen_manips = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38]#, 39, 40, 42, 43, 44]



##






def nice_plot(xpos,ypos,ax=None,xerr=None,yerr=None,xlabel="",ylabel="",
              annotate=None,old_data=False,
              color="b",error_bar_color="k",marker_size=3,capsize=error_bar_width,
              marker="d",solid_marker=solid_marker,in_solid=None,solid_color=solid_solid_color):
    if ax is None:
        fig, ax = plt.subplots()
    if old_data:
        ax.errorbar(old_data[0],old_data[1],xerr=old_data[2],yerr=old_data[3],
                    fmt=" ",capsize=capsize,color=color,ecolor=error_bar_color,elinewidth=.5, alpha=error_bar_alpha,markeredgewidth=markeredgewidth)
        ax.scatter(old_data[0],old_data[1],color=color,s=marker_size,marker=marker,zorder=5,edgecolors="k",linewidth=0.01)

    # all data
    ax.errorbar(xpos,ypos, xerr=xerr,yerr=yerr,
                fmt=" ",capsize=capsize,color=color,ecolor=error_bar_color,elinewidth=.5,alpha=error_bar_alpha,markeredgewidth=markeredgewidth)
    ax.scatter(xpos,ypos,color=color,s=marker_size,marker=marker,zorder=5,edgecolors="k",linewidth=0.01)

    # solids
    if not in_solid is None:
        ax.errorbar(xpos[in_solid],ypos[in_solid],
        xerr=xerr[...,in_solid],yerr=yerr[...,in_solid],
        fmt=" ",capsize=capsize,
        color=solid_color,ecolor=error_bar_color,
        elinewidth=.5,alpha=error_bar_alpha ,
        markeredgewidth=markeredgewidth)

        ax.scatter(xpos[in_solid],ypos[in_solid],
        color=solid_color,s=marker_size,
        marker=solid_marker,zorder=5,
        edgecolors="k",linewidth=0.01)
    if not annotate is None:
        for i in range(len(annotate)):
            ax.annotate(annotate[i], (xpos[i], ypos[i]))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)





#
bins_2 = np.array([-1.1,-0.5,0,1,2])
bins = np.array([-2,-0.5,0,1,5])

solid = np.array(solid_per_manip_2)
not_solid = np.logical_not(solid)

bins_bool = np.array( [ np.logical_and(
                            np.logical_and( lc_per_manip_2<bins[i+1],
                                            lc_per_manip_2>bins[i]),
                            not_solid)
                        for i in range(len(bins)-1)] )


bins_bool_solid = solid

sig_binned = np.array([np.mean(sigma_yy_0_tip_per_manip_2[b]) for b in bins_bool])
sig_binned_std = np.array([np.std(sigma_yy_0_tip_per_manip_2[b]) for b in bins_bool])

sig_sol = np.mean(sigma_yy_0_tip_per_manip_2[bins_bool_solid])
sig_sol_std = np.std(sigma_yy_0_tip_per_manip_2[bins_bool_solid])


















### plot 6a.

fig,axes = plt.subplots(1,3,sharex=False)
fig.subplots_adjust(left=0.07, bottom=0.19, right=0.999, top=0.96, wspace=0.35)
fig.set_size_inches(size_fig_ell_triple)

ax_0,ax_1,ax_3 = axes


ax_0.errorbar(lc_solid_mean,ell_per_lc_round_solid*1000/30,yerr = ell_per_lc_round_width_solid*1000/30,
                 fmt=" ",capsize=error_bar_width,color="k",
               ecolor=error_bar_color,
               elinewidth=.5,alpha=error_bar_alpha,markeredgewidth=markeredgewidth)

ax_0.scatter((lc_list_round[:-1]+lc_list_round[1:])/2,ell_per_lc_round*1000/30,
        c=main_plot_color,s=scatter_size*2,
        marker="d",zorder=2,
        edgecolors="k",linewidth=0.01)



ax_0.scatter(lc_solid_mean,ell_per_lc_round_solid*1000/30,
        c=solid_solid_color,s=scatter_size*4/3,
        marker="o",zorder=2,
        edgecolors="k",linewidth=0.01)



ax_0.errorbar((lc_list_round[:-1]+lc_list_round[1:])/2,ell_per_lc_round*1000/30,yerr = ell_per_lc_round_width*1000/30,
                 fmt=" ",capsize=error_bar_width,color="k",
               ecolor=error_bar_color,
               elinewidth=.5,alpha=error_bar_alpha,markeredgewidth=markeredgewidth)



#ax_0.legend()
ax_0.set_xlim(-1.1,2)

ax_0.set_ylim(0.8,4.3)

ax_0.set_xlabel("${}$".format(LC_name_short))
ax_0.set_ylabel(r"$\left\langle\ell_{slip}\right\rangle\,/\,\ell_{eye}$")

ax_0.xaxis.set_minor_locator(MultipleLocator(0.5))
ax_0.xaxis.set_major_locator(MultipleLocator(1))
ax_0.yaxis.set_minor_locator(MultipleLocator(0.5))
ax_0.yaxis.set_major_locator(MultipleLocator(1))


temp = np.cumsum(np.diff(lc_event)!=0)
temp = np.insert(temp,0,0)
mean_slip = np.array(  [
                            np.mean(
                                    ell[temp == i]
                                    )
                            for i in range(max(temp)+1)
                       ]
                    )


ax_0.scatter(lc_event,
                ell/30,
                marker = scatter_marker, s = scatter_size/4,
                c=secondary_plot_color,alpha = secondary_plot_alpha ,
                zorder=-1,edgecolors="k",linewidth=0)



### plot 6b.

loaded_data = np.load(main_folder + "data_for_figures/figure_4.npy",allow_pickle=True).all()
locals().update(loaded_data)


mean_lc=data_loaded["mean_lc"]
manip_num=np.array(data_loaded["manip_num"])
LC_here = mean_lc[manip_num==chosen_manip_fig41][0]

index_start= np.array(data_loaded_3["start_per_event"])
loading_contrast = np.array(data_loaded_3["lc_per_event"])
from_solid=np.array(data_loaded_3["solid_per_event"])

x=np.array([1,2,3,4,5,8,11,12,13,14])*1e-2 -0.005

mean_lc=data_loaded_2["mean_lc"]
manip_num=np.array(data_loaded_2["manip_num"])
LC_there = mean_lc[manip_num==chosen_manip_fig41_sec][0]

def my_mean(x):
    m = 0
    c = 0
    for xi in x:
        if xi!= 0:
            m+=xi
            c+=1
    if m==0:
        return(np.nan)
    return(m/c)

nuc_pt = x[index_start]
ell_nuc = 1000*(np.abs(nuc_pt-0.075))

ell_nuc_2 = nuc_pt*(nuc_pt>0.075)
ell_nuc_3 = nuc_pt*(nuc_pt<0.075)

xxs = np.arange(0,15,1)*10

lc_per_ell_nuc = [np.mean(loading_contrast[ell_nuc == xxs[i]]) if len(loading_contrast[ell_nuc == xxs[i]])!=0 else np.nan for i in range(15)]

lc_list_nuc = np.array([-1,-0.5,0,1,2])


ell_per_lc_nuc = np.array(
        [np.mean(
            ell_nuc[np.logical_and(
                loading_contrast<lc_list_nuc[i+1] ,
                loading_contrast>lc_list_nuc[i])
                ]
            )/1000
        if len(
            ell_nuc[np.logical_and(
                loading_contrast<lc_list_nuc[i+1],
                loading_contrast>lc_list_nuc[i] ,
                np.logical_not(from_solid)
                )
                ]
            )!=0
        else np.nan
        for i in range(len(lc_list_nuc)-1)]
        )


ell_per_lc_nuc_width = np.array(
        [np.std(
            ell_nuc[np.logical_and(
                loading_contrast<lc_list_nuc[i+1] ,
                loading_contrast>lc_list_nuc[i] ,
                np.logical_not(from_solid)
                )
            ])/1000
        if len(
            ell_nuc[np.logical_and(
                loading_contrast<lc_list_nuc[i+1],
                loading_contrast>lc_list_nuc[i])
                ]
            )!=0
        else np.nan
        for i in range(len(lc_list_nuc)-1)]
        )


ell_per_lc_nuc_solid = np.mean(ell_nuc[from_solid])/1000

ell_per_lc_nuc_width_solid = np.std(ell_nuc[from_solid])/1000

lc_solid_mean = np.mean(loading_contrast[from_solid])

ax_1.scatter((lc_list_nuc[:-1]+lc_list_nuc[1:])/2,ell_per_lc_nuc*1000/15,
        c=main_plot_color,s=scatter_size*2,
        marker="d",zorder=3,
        edgecolors="k",linewidth=0.01)


ax_1.errorbar((lc_list_nuc[:-1]+lc_list_nuc[1:])/2,ell_per_lc_nuc*1000/15,yerr = ell_per_lc_nuc_width*1000/15,
                 fmt=" ",capsize=error_bar_width,color="k",
               ecolor=error_bar_color,
               elinewidth=.5,alpha=error_bar_alpha,markeredgewidth=markeredgewidth)


ax_1.scatter(lc_solid_mean,ell_per_lc_nuc_solid*1000/15,
        c=solid_solid_color,s=scatter_size*4/3,
        marker="o",zorder=3,
        edgecolors="k",linewidth=0.01)


ax_1.errorbar(lc_solid_mean,ell_per_lc_nuc_solid*1000/15,yerr = ell_per_lc_nuc_width_solid*1000/15,
               fmt=" ",capsize=error_bar_width,color="k",
               ecolor=error_bar_color,
               elinewidth=.5,alpha=error_bar_alpha,markeredgewidth=markeredgewidth)


ax_1.scatter(loading_contrast,
                ell_nuc/15,
                marker = scatter_marker, s = scatter_size/4,
                c=secondary_plot_color,alpha = secondary_plot_alpha ,
                zorder=-1,edgecolors="k",linewidth=0)


ax_1.xaxis.set_minor_locator(MultipleLocator(0.5))
ax_1.xaxis.set_major_locator(MultipleLocator(1))
ax_1.yaxis.set_minor_locator(MultipleLocator(0.5))
ax_1.yaxis.set_major_locator(MultipleLocator(1))



ax_1.set_xlim(-1.1,2)
ax_1.set_ylim(0.8,4.3)
ax_1.set_xlabel("${}$".format(LC_name_short))
ax_1.set_ylabel(r"${\left\langled_{nuc}\right\rangle}\;/\;({\ell_{eye}\,/\,2})$")

ax_1.set_xticks([-1,0,1,2])
ax_1.set_xticklabels(["-1","0","1","2"])
ax_1.yaxis.set_minor_locator(MultipleLocator(0.5))
ax_1.yaxis.set_major_locator(MultipleLocator(1))















### plot 6b.
"""

nice_plot(bins_2[1:]/2+bins_2[:-1]/2,sig_binned/1e6,
              ax=ax_2,xerr=None,yerr=sig_binned_std/1e6,
              color=main_plot_color, marker_size=scatter_size*2,
              marker="d",solid_marker="o",in_solid=None,
              error_bar_color=error_bar_color,
              solid_color=solid_solid_color)



nice_plot(lc_solid_mean,sig_sol/1e6,
              ax=ax_2,xerr=None,yerr=sig_sol_std/1e6,
              color=solid_solid_color, marker_size=scatter_size*4/3,
              marker="o",solid_marker="o",in_solid=None,
              error_bar_color=error_bar_color,
              solid_color=solid_solid_color)





ax_2.set_xlabel("${}$".format(LC_name_short))
ax_2.set_ylabel(r"$\left\langle\sigma_{yy}^{0}(x_{nuc})\right\rangle$ (MPa)")
ax_2.set_xlim([-1.1,2])
ax_2.set_ylim([0,3.8])
ax_2.xaxis.set_minor_locator(MultipleLocator(0.5))
ax_2.xaxis.set_major_locator(MultipleLocator(1))
ax_2.yaxis.set_minor_locator(MultipleLocator(0.5))
ax_2.yaxis.set_major_locator(MultipleLocator(1))





ax_2.scatter(lc_per_manip_2,
                sigma_yy_0_tip_per_manip_2/1e6,
                marker = scatter_marker, s = scatter_size/4,
                c=secondary_plot_color,alpha = secondary_plot_alpha ,
                zorder=-1,edgecolors="k",linewidth=0)

ax_2.scatter(lc_per_manip_2[solid_per_manip_2],
                sigma_yy_0_tip_per_manip_2[solid_per_manip_2]/1e6,
                marker = 'o', s = scatter_size/4*2/3,
                c=secondary_plot_color,alpha = secondary_plot_alpha ,
                zorder=-1,edgecolors="k",linewidth=0)


"""











###


ax_3.errorbar(ell_per_lc_nuc*1000/15,
        ell_per_lc_round*1000/30,
        xerr = ell_per_lc_nuc_width*1000/15,
        yerr = ell_per_lc_round_width*1000/30,
        fmt=" ",capsize=error_bar_width,color="k",
        ecolor=error_bar_color,
        elinewidth=.5,alpha=error_bar_alpha,markeredgewidth=markeredgewidth)


ax_3.scatter(ell_per_lc_nuc*1000/15,
        ell_per_lc_round*1000/30,
        c=main_plot_color,s=scatter_size*2,
        marker="d",zorder=2,
        edgecolors="k",linewidth=0.01)


ax_3.errorbar(ell_per_lc_nuc_solid*1000/15,
        ell_per_lc_round_solid*1000/30,
        xerr = ell_per_lc_nuc_width_solid*1000/15,
        yerr = ell_per_lc_round_width_solid*1000/30,
        fmt=" ",capsize=error_bar_width,color="k",
        ecolor=error_bar_color,
        elinewidth=.5,alpha=error_bar_alpha,markeredgewidth=markeredgewidth)


ax_3.scatter(ell_per_lc_nuc_solid*1000/15,
        ell_per_lc_round_solid*1000/30,
        c=solid_solid_color,s=scatter_size*4/3,
        marker="o",zorder=2,
        edgecolors="k",linewidth=0.01)



ax_3.plot([0,4.3],[0,4.3],linestyle = '--',color = "k")


#ax_3.legend()
ax_3.set_xlim(0.8,4.3)
ax_3.set_ylim(0.8,4.3)
#ax_3.set_ylim(23,120)
ax_3.set_xlabel(r"${\left\langled_{nuc}\right\rangle}\;/\;({\ell_{eye}\,/\,2})$")
ax_3.set_ylabel(r"$\left\langle\ell_{slip}\right\rangle\,/\,\ell_{eye}$")

ax_3.xaxis.set_minor_locator(MultipleLocator(0.5))
ax_3.xaxis.set_major_locator(MultipleLocator(1))
ax_3.yaxis.set_minor_locator(MultipleLocator(0.5))
ax_3.yaxis.set_major_locator(MultipleLocator(1))
#ax_3.yaxis.set_minor_locator(MultipleLocator(25))
#ax_3.yaxis.set_major_locator(MultipleLocator(50))



set_grid(axes)

y=0.93
fig.text(0.01,y,"a.",size=LETTER_SIZE, weight='bold')
fig.text(0.34,y,"b.",size=LETTER_SIZE, weight='bold')
fig.text(0.68,y,"c.",size=LETTER_SIZE, weight='bold')



fig.savefig(main_folder + "Figures_chap_article/ell_triple.png",dpi=dpi_global)
fig.savefig(main_folder + "Figures_chap_article/ell_triple.pdf")
fig.savefig(main_folder + "Figures_chap_article/ell_triple.svg")
plt.close('all')


