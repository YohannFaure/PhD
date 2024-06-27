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










### Fig 6 b.

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

fig,axes = plt.subplots(1,2,sharex=True)
fig.subplots_adjust(left=0.1, bottom=0.2, right=0.99, top=0.97, wspace=0.3)
fig.set_size_inches(size_fig_article_6_backup)

axes[0].errorbar(lc_solid_mean,ell_per_lc_round_solid*1000/30,yerr = ell_per_lc_round_width_solid*1000/30,
                 fmt=" ",capsize=error_bar_width,color="k",
               ecolor=error_bar_color,
               elinewidth=.5,alpha=error_bar_alpha,markeredgewidth=markeredgewidth)

axes[0].scatter((lc_list_round[:-1]+lc_list_round[1:])/2,ell_per_lc_round*1000/30,
        c=main_plot_color,s=scatter_size*2,
        marker="d",zorder=2,
        edgecolors="k",linewidth=0.01)



axes[0].scatter(lc_solid_mean,ell_per_lc_round_solid*1000/30,
        c=solid_solid_color,s=scatter_size*4/3,
        marker="o",zorder=2,
        edgecolors="k",linewidth=0.01)



axes[0].errorbar((lc_list_round[:-1]+lc_list_round[1:])/2,ell_per_lc_round*1000/30,yerr = ell_per_lc_round_width*1000/30,
                 fmt=" ",capsize=error_bar_width,color="k",
               ecolor=error_bar_color,
               elinewidth=.5,alpha=error_bar_alpha,markeredgewidth=markeredgewidth)



#axes[0].legend()
axes[0].set_xlim(-1.1,2)
#axes[0].set_ylim(23,120)
axes[0].set_xlabel("${}$".format(LC_name_short))
axes[0].set_ylabel(r"$\left\langle\ell_{slip}\right\rangle\,/\,\ell_{eye}$")

axes[0].xaxis.set_minor_locator(MultipleLocator(0.5))
axes[0].xaxis.set_major_locator(MultipleLocator(1))
axes[0].yaxis.set_minor_locator(MultipleLocator(0.5))
axes[0].yaxis.set_major_locator(MultipleLocator(1))





"""
mean_lc                         = data_fig_5["mean_lc"]
lc_event                        = data_fig_5["lc_event"]
ell                             = data_fig_5["ell"]
is_solid                        = data_fig_5["is_solid"]
"""


temp = np.cumsum(np.diff(lc_event)!=0)
temp = np.insert(temp,0,0)
mean_slip = np.array(  [
                            np.mean(
                                    ell[temp == i]
                                    )
                            for i in range(max(temp)+1)
                       ]
                    )

"""
axes[0].scatter(mean_lc,
                mean_slip/30,
                marker = scatter_marker, s = scatter_size/4,
                c=secondary_plot_color,alpha = secondary_plot_alpha ,
                zorder=-1,edgecolors="k",linewidth=0)

axes[0].scatter(mean_lc[is_solid],
                mean_slip[is_solid]/30,
                marker = 'o', s = scatter_size/4*2/3,
                c=secondary_plot_color,alpha = secondary_plot_alpha ,
                zorder=-1,edgecolors="k",linewidth=0)
"""



axes[0].scatter(lc_event,
                ell/30,
                marker = scatter_marker, s = scatter_size/4,
                c=secondary_plot_color,alpha = secondary_plot_alpha ,
                zorder=-1,edgecolors="k",linewidth=0)










### plot 6b.


nice_plot(bins_2[1:]/2+bins_2[:-1]/2,sig_binned/1e6,
              ax=axes[1],xerr=None,yerr=sig_binned_std/1e6,
              color=main_plot_color, marker_size=scatter_size*2,
              marker="d",solid_marker="o",in_solid=None,
              error_bar_color=error_bar_color,
              solid_color=solid_solid_color)



nice_plot(lc_solid_mean,sig_sol/1e6,
              ax=axes[1],xerr=None,yerr=sig_sol_std/1e6,
              color=solid_solid_color, marker_size=scatter_size*4/3,
              marker="o",solid_marker="o",in_solid=None,
              error_bar_color=error_bar_color,
              solid_color=solid_solid_color)





axes[1].set_xlabel("${}$".format(LC_name_short))
axes[1].set_ylabel(r"$\left\langle\sigma_{yy}^{0}(x_{nuc})\right\rangle$ (MPa)")
axes[1].set_xlim([-1.1,2])
axes[1].set_ylim([0,3.8])
axes[1].xaxis.set_minor_locator(MultipleLocator(0.5))
axes[1].xaxis.set_major_locator(MultipleLocator(1))
axes[1].yaxis.set_minor_locator(MultipleLocator(0.5))
axes[1].yaxis.set_major_locator(MultipleLocator(1))
set_grid(axes)








"""
axin = fig.add_axes([0.6447, 0.295,   0.27,  0.23],facecolor="w")



axin.scatter(lc_per_manip_2,
                sigma_yy_0_tip_per_manip_2/1e6,
                marker = scatter_marker, s = scatter_size/4,
                c=main_plot_color,alpha = 1 ,
                zorder=1,edgecolors="k",linewidth=0.01)

axin.scatter(lc_per_manip_2[solid_per_manip_2],
                sigma_yy_0_tip_per_manip_2[solid_per_manip_2]/1e6,
                marker = 'o', s = scatter_size/4*2/3,
                c=solid_solid_color,alpha = 1 ,
                zorder=2,edgecolors="k",linewidth=0.01)



axin.set_xlim([-1.1,2])
axin.set_ylim([0,4])
#axin.set_xlabel("${}$".format(LC_name_short))
#axin.set_ylabel("$\sigma_{yy}^{0}(x_{nuc})$ (MPa)")

axin.xaxis.set_minor_locator(MultipleLocator(1))
axin.xaxis.set_major_locator(MultipleLocator(2))
axin.yaxis.set_minor_locator(MultipleLocator(1))
axin.yaxis.set_major_locator(MultipleLocator(2))


set_up_inset(axin)
"""

axes[-1].scatter(lc_per_manip_2,
                sigma_yy_0_tip_per_manip_2/1e6,
                marker = scatter_marker, s = scatter_size/4,
                c=secondary_plot_color,alpha = secondary_plot_alpha ,
                zorder=-1,edgecolors="k",linewidth=0)

axes[-1].scatter(lc_per_manip_2[solid_per_manip_2],
                sigma_yy_0_tip_per_manip_2[solid_per_manip_2]/1e6,
                marker = 'o', s = scatter_size/4*2/3,
                c=secondary_plot_color,alpha = secondary_plot_alpha ,
                zorder=-1,edgecolors="k",linewidth=0)


y=0.94
fig.text(0.01,y,"a.",size=LETTER_SIZE, weight='bold')
fig.text(0.5,y,"b.",size=LETTER_SIZE, weight='bold')



fig.savefig(main_folder + "Figures_chap_article/figure_6.png",dpi=dpi_global)
fig.savefig(main_folder + "Figures_chap_article/figure_6.pdf")
fig.savefig(main_folder + "Figures_chap_article/figure_6.svg")
plt.close('all')





