main_folder = 'E:/manuscrit_these/'
main_folder = '../'


with open(main_folder + "parameters.py", 'r') as f:
        exec(f.read())








def nice_plot(xpos,ypos,ax=None,xerr=None,yerr=None,xlabel="",ylabel="",
              annotate=None,old_data=False,
              color="b",error_bar_color="k",marker_size=3,capsize=3,
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
    ax.errorbar(xpos[in_solid],ypos[in_solid], xerr=xerr[...,in_solid],yerr=yerr[...,in_solid],
                fmt=" ",capsize=capsize,color=solid_color,ecolor=error_bar_color,elinewidth=.5,alpha=error_bar_alpha ,markeredgewidth=markeredgewidth)
    ax.scatter(xpos[in_solid],ypos[in_solid],color=solid_color,s=marker_size,marker=solid_marker,zorder=5,edgecolors="k",linewidth=0.01)
    if not annotate is None:
        for i in range(len(annotate)):
            ax.annotate(annotate[i], (xpos[i], ypos[i]))

    ax.set_xlabel(xlabel,labelpad=2)
    ax.set_ylabel(ylabel)











n_manip=44
to_remove = [7,19,20,21,22,23,24,25,32,41]

solids = [14,15,16,17,18,37,38]
weird = [32,33]
manips = [i for i in range(1,n_manip+1) if i not in to_remove]


in_solid = np.array([m in solids for m in manips])

##

dicto_save = np.load(main_folder + "data_for_figures/creep_5_paires.npy",allow_pickle=True).all()
locals().update(dicto_save)


##
ieslip=np.array([[creep_left_lefts[i][-1],creep_left_centers[i][-1],creep_center[i]/100,creep_right_centers[i][-1],creep_right_rights[i][-1]] for i in range(len(creep_left_lefts))])


ieslip_err=np.array([[[min(creep_left_lefts[i][-5:]),max(creep_left_lefts[i][-5:])],
                    [min(creep_left_centers[i][-5:]),max(creep_left_centers[i][-5:])],
                    [(creep_center[i]-sigma_creep_center[i])/100, (creep_center[i]+sigma_creep_center[i])/100],
                    [min(creep_right_centers[i][-5:]),max(creep_right_centers[i][-5:])],
                    [min(creep_right_rights[i][-5:]),max(creep_right_rights[i][-5:])]
                    ] for i in range(len(creep_left_lefts))])

ieslip_err=np.rollaxis(ieslip_err,2,0)

ieslip_err=np.abs(ieslip_err-ieslip)




###

fig,axes=plt.subplots(1,5,sharex=True,sharey=True,width_ratios=[1,1,1,1,1])
plt.subplots_adjust(left=0.075, right=0.99, bottom=0.17, top=0.85, wspace=0.1, hspace=0.4)


fig.set_size_inches(size_fig_creep_5_paires)
for i in range(5):
    if i==2:
        nice_plot(mean_lc,ieslip[:,i],ax=axes[i],
            xerr=wide_lc, yerr=ieslip_err[:,:,i],
            xlabel="${}$".format(LC_name_short), ylabel=None,
            in_solid=in_solid,
            color=granular_color, error_bar_color=error_bar_color,
            marker_size=scatter_size, capsize=error_bar_width, marker=scatter_marker,
            solid_color = hole_in_solid_color)
    else :
        nice_plot(mean_lc,ieslip[:,i],ax=axes[i],
            xerr=wide_lc, yerr=ieslip_err[:,:,i],
            xlabel=None, ylabel=None,
            in_solid=np.array([m in solids for m in manips]),
            color=solid_in_granular_color, error_bar_color=error_bar_color,
            marker_size=scatter_size, capsize=error_bar_width, marker=scatter_marker,
            solid_color = solid_in_granular_color)






#
# nice_plot(mean_lc,creep_center/100-creep_sides/100,ax=axes[-1],
#         xerr=wide_lc,yerr=sigma_creep_center/100,
#         xlabel="${}$".format(LC_name_short),ylabel="$S_f^{hole}-S_f^{solid}$",
#         in_solid=in_solid,
#         color=main_plot_color, error_bar_color=error_bar_color,
#         marker_size=scatter_size, capsize=error_bar_width, marker=scatter_marker )
#











#axes[-1].grid(which="both")










# adjust ticks
# Top Left
axes[0].set_xlim([-1.1,2])
axes[0].set_xticks([-1,0,1,2])
axes[0].set_xticklabels(["-1","0","1","2"])
axes[0].set_ylim([-4/100,28/100])
axes[0].set_yticks([0,10/100,20/100])
axes[0].set_yticklabels(["0",0.1,0.2])
axes[0].xaxis.set_minor_locator(MultipleLocator(1))
axes[0].yaxis.set_minor_locator(MultipleLocator(0.05))

#plt.setp(ax0.get_xticklabels(), visible=False)


set_grid(axes)




axes[0].set_title("solide\n$x\simeq10$ mm",size=INSET_MEDIUM_SIZE,pad=4)
axes[1].set_title("solide\n$x\simeq35$ mm",size=INSET_MEDIUM_SIZE,pad=4)
axes[2].set_title("œil\n$x=75$ mm",size=INSET_MEDIUM_SIZE,pad=4)
axes[3].set_title("solide\n$x\simeq115$ mm",size=INSET_MEDIUM_SIZE,pad=4)
axes[4].set_title("solide\n$x\simeq140$ mm",size=INSET_MEDIUM_SIZE,pad=4)
#axes[-1].set_title("$S_f^{hole}-S_f^{solid}$",size=INSET_MEDIUM_SIZE)

axes[0].set_ylabel(r"$S_{f}^{\{\ell\}}$")




plt.savefig(main_folder + "Figures_chap_article/creep_5_paires.png",dpi=dpi_global)
plt.savefig(main_folder + "Figures_chap_article/creep_5_paires.pdf")
plt.savefig(main_folder + "Figures_chap_article/creep_5_paires.svg")

plt.close('all')















