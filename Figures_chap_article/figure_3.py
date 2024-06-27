main_folder = 'E:/manuscrit_these/'
main_folder = '../'


with open(main_folder + "parameters.py", 'r') as f:
        exec(f.read())



loaded_data = np.load(main_folder + "data_for_figures/figure_3.npy",allow_pickle=True).all()
locals().update(loaded_data)


###

def nice_plot(xpos,ypos,ax=None,xerr=None,yerr=None,xlabel="",ylabel="",
              annotate=None,old_data=False,
              color="b",error_bar_color="k",marker_size=3,capsize=3,
              marker="d",solid_marker=solid_marker,in_solid=None,
              solid_color=solid_solid_color,
              label = None,
              label_solid = None):
    if ax is None:
        fig, ax = plt.subplots()
    if old_data:
        ax.errorbar(old_data[0],old_data[1],xerr=old_data[2],yerr=old_data[3],
                    fmt=" ",capsize=capsize,color=color,ecolor=error_bar_color,elinewidth=.5, alpha=error_bar_alpha,markeredgewidth=markeredgewidth)
        ax.scatter(old_data[0],old_data[1],color=color,s=marker_size,marker=marker,zorder=5,edgecolors="k",linewidth=0.01)

    # all data
    ax.errorbar(xpos,ypos, xerr=xerr,yerr=yerr,
                fmt=" ",capsize=capsize,color=color,ecolor=error_bar_color,elinewidth=.5,alpha=error_bar_alpha,markeredgewidth=markeredgewidth)
    ax.scatter(xpos,ypos,color=color,s=marker_size,marker=marker,zorder=5,edgecolors="k",linewidth=0.01,label = label)

    # solids
    ax.errorbar(xpos[in_solid],ypos[in_solid], xerr=xerr[...,in_solid],yerr=yerr[...,in_solid],
                fmt=" ",capsize=capsize,color=solid_color,ecolor=error_bar_color,elinewidth=.5,alpha=error_bar_alpha ,markeredgewidth=markeredgewidth)
    ax.scatter(xpos[in_solid],ypos[in_solid],color=solid_color,s=marker_size,marker=solid_marker,zorder=5,edgecolors="k",linewidth=0.01,label = label_solid)
    if not annotate is None:
        for i in range(len(annotate)):
            ax.annotate(annotate[i], (xpos[i], ypos[i]))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)



substract_sides = False
plot_sides = True
plot_old = True





##


fig = plt.figure(layout=None)
fig.set_size_inches(size_fig_article_3)

gs = fig.add_gridspec(nrows=8, ncols=11, left=0.13, right=0.99,
                      hspace=0, wspace=0.8)
ax0 = fig.add_subplot(gs[0:4, 0:4])
ax1 = fig.add_subplot(gs[4:, 0:4], sharex=ax0,sharey=ax0)
ax2 = fig.add_subplot(gs[0:8,5: ], sharex=ax0,sharey=ax0)


ax3 = fig.add_axes([0.63, 0.75,   0.22,  0.22],facecolor="w")

axes=np.array([ax0,ax1,ax2,ax3])




if plot_old:
    old_data0=[old_data["lc"],old_data["ie_slip_grains"]/100,
            old_data["lc_err"],old_data["ie_slip_err"]/100]
    old_data1=[old_data["lc"],old_data["ie_slip_solid"]/100,
            old_data["lc_err"],old_data["ie_slip_err"]/100]
    old_data2=[old_data["lc"],old_data["ie_slip_grains"]/100-old_data["ie_slip_solid"]/100,
            old_data["lc_err"],old_data["ie_slip_err"]/100]
    old_data3=[old_data["ie_slip_grains"]/100-old_data["ie_slip_solid"]/100,old_data["freq"],
                old_data["ie_slip_err"]/100,old_data["freq_err"]]

else:
    old_data0=None
    old_data1=None
    old_data2=None
    old_data3=None


if True:
    in_solid=np.isin(new_data["manip_num"],solids)
else:
    in_solid=None


nice_plot(new_data["mean_lc"],new_data["creep_center"]/100,ax=ax0,
        xerr=new_data["wide_lc"],yerr=new_data["sigma_creep_center"]/100,
        xlabel=None,ylabel="$S^{eye}_f$",
        old_data=old_data0,in_solid=in_solid,
        color=granular_color, error_bar_color=error_bar_color,
        marker_size=scatter_size, capsize=error_bar_width, marker=scatter_marker,
        solid_color = hole_in_solid_color,
        label = "granulaire",
        label_solid = "à vide")


nice_plot(new_data["mean_lc"],new_data["creep_sides"]/100,ax=ax1,
        xerr=new_data["wide_lc"],yerr=new_data["sigma_creep_center"]/100,
        xlabel=r"${}$".format(LC_name_short),ylabel="$S^{solid}_f$",
        old_data=old_data1,in_solid=in_solid,
        color=solid_in_granular_color, error_bar_color=error_bar_color,
        marker_size=scatter_size, capsize=error_bar_width, marker=scatter_marker,
        solid_color=solid_in_granular_color,
        label = "granulaire",
        label_solid = "à vide")


nice_plot(new_data["mean_lc"],new_data["creep_center"]/100-new_data["creep_sides"]/100,ax=ax2,
        xerr=new_data["wide_lc"],yerr=new_data["sigma_creep_center"]/100,
        xlabel=r"${}$".format(LC_name_short),ylabel=r"$S^{eye}_f-S^{solid}_f$",
        old_data=old_data2,in_solid=in_solid,
        color=main_plot_color, error_bar_color=error_bar_color,
        marker_size=scatter_size, capsize=error_bar_width, marker=scatter_marker,
        label = "granulaire",
        label_solid = "à vide")


nice_plot(new_data["creep_center"]/100-new_data["creep_sides"]/100,new_data["mean_freq"],ax=ax3,
            xerr=new_data["sigma_creep_center"]/100,yerr=new_data["wide_freq"],
            xlabel=r"$S^{eye}_f-S^{solid}_f$" ,ylabel=freq_name_short ,
            old_data=old_data3,in_solid=in_solid,
            color=main_plot_color, error_bar_color=error_bar_color,
            marker_size=scatter_size/4, capsize=error_bar_width/2, marker="d")

legends = [["œil (vide)" , "solide"],
           ["œil (rempli)", "solide"]]

#get handles and labels
#handles, labels = ax1.get_legend_handles_labels()
#order = [0]
#ax1.legend([handles[idx] for idx in order],[labels[idx] for idx in order],prop={'size': INSET_SMALL_SIZE},framealpha = 0, facecolor = (1,1,1,0),loc = "upper left",bbox_to_anchor=(-.07,1.),handletextpad=0.1)

handles, labels = ax0.get_legend_handles_labels()
order = [1,0]
ax0.legend([handles[idx] for idx in order],[labels[idx] for idx in order],prop={'size': INSET_SMALL_SIZE},framealpha = 0, facecolor = (1,1,1,0),loc = "upper left",bbox_to_anchor=(-.07,1.045),handletextpad=0.1)

handles, labels = ax1.get_legend_handles_labels()
order = [1,0]
ax1.legend([handles[idx] for idx in order],[labels[idx] for idx in order],prop={'size': INSET_SMALL_SIZE},framealpha = 0, facecolor = (1,1,1,0),loc = "upper left",bbox_to_anchor=(-.07,1.0),handletextpad=0.1)


handles, labels = ax2.get_legend_handles_labels()
order = [1,0]
ax2.legend([handles[idx] for idx in order],[labels[idx] for idx in order],prop={'size': INSET_SMALL_SIZE},framealpha = 0, facecolor = (1,1,1,0),loc = "lower right",bbox_to_anchor=(1.01,0),handletextpad=0.1)



ax0.get_yaxis().labelpad=1
ax1.get_yaxis().labelpad=1
ax2.get_yaxis().labelpad=0
ax1.get_xaxis().labelpad=2
ax2.get_xaxis().labelpad=2







# adjust ticks
# Top Left
ax0.set_xlim([-1.1,2.8])
ax0.set_xticks([-1,0,1,2])
ax0.set_xticklabels(["-1","0","1","2"])
ax0.set_ylim([-4/100,28/100])
ax0.set_yticks([0,10/100,20/100])
ax0.set_yticklabels(["0",0.1,0.2])
ax0.xaxis.set_minor_locator(MultipleLocator(1))
ax0.yaxis.set_minor_locator(MultipleLocator(0.05))


plt.setp(ax0.get_xticklabels(), visible=False)


# right
ax3.set_xlim([-2/100,18/100])
ax3.set_xticks([0,10/100])#ax3.set_ylim([0,0.25])
ax3.set_xticklabels(["0",0.1])
ax3.xaxis.set_minor_locator(MultipleLocator(0.05))

ax3.set_ylim([-0.01,0.6])
ax3.set_yticks([0,0.5])
ax3.set_yticklabels(["0",0.5])
ax3.yaxis.set_minor_locator(MultipleLocator(0.25))
set_up_inset(ax3)

ax3.get_yaxis().labelpad=inset_label_pad-2
ax3.get_xaxis().labelpad=inset_label_pad


set_grid(axes[-1])
set_grid(axes[:-1])

axes[-1].grid(False,which="both")




y=0.95
fig.text(0.01,y,"a.",size=LETTER_SIZE, weight='bold')
#fig.text(0.01,y/1.9,"b",size=LETTER_SIZE, weight='bold')
fig.text(.43,y,"b.",size=LETTER_SIZE, weight='bold')
# save


fig.subplots_adjust( top=0.98, bottom=0.12)

plt.savefig(main_folder + "Figures_chap_article/figure_3.png",dpi=dpi_global)
plt.savefig(main_folder + "Figures_chap_article/figure_3.pdf")
plt.savefig(main_folder + "Figures_chap_article/figure_3.svg")



plt.close('all')


### Figure pour Elsa

fig,ax=plt.subplots(1)
fig.set_size_inches((90*mm,70*mm))

nice_plot(new_data["mean_lc"],new_data["mean_freq"],ax=ax,
            xerr=new_data["wide_lc"],yerr=new_data["wide_freq"],
            xlabel=r"${}$".format(LC_name_short) ,ylabel=freq_name_short ,
            old_data=False,in_solid=in_solid,
            color=main_plot_color, error_bar_color=error_bar_color,
            marker_size=scatter_size, capsize=error_bar_width, marker="d")


plt.tight_layout()

ax.xaxis.set_minor_locator(MultipleLocator(.5))
ax.xaxis.set_major_locator(MultipleLocator(1))

ax.yaxis.set_minor_locator(MultipleLocator(0.05))
ax.yaxis.set_major_locator(MultipleLocator(0.1))

ax.set_xlim([-1.1,2.1])

set_grid(ax)


plt.savefig(main_folder + "Figures_chap_article/figures_pour_elsa/freq_vs_c_sigma.png",dpi=dpi_global)
plt.savefig(main_folder + "Figures_chap_article/figures_pour_elsa/freq_vs_c_sigma.pdf")
plt.savefig(main_folder + "Figures_chap_article/figures_pour_elsa/freq_vs_c_sigma.svg")
plt.close('all')













fig,ax=plt.subplots(1)
fig.set_size_inches((90*mm,70*mm))

nice_plot(new_data["mean_lc"],new_data["creep_center"]/100-new_data["creep_sides"]/100,ax=ax,
        xerr=new_data["wide_lc"],yerr=new_data["sigma_creep_center"]/100,
        xlabel=r"${}$".format(LC_name_short),ylabel=r"$S^{eye}_f-S^{solid}_f$",
        old_data=old_data2,in_solid=in_solid,
        color=main_plot_color, error_bar_color=error_bar_color,
        marker_size=scatter_size, capsize=error_bar_width, marker=scatter_marker,
        label = "granulaire",
        label_solid = "à vide")



ax.xaxis.set_minor_locator(MultipleLocator(.5))
ax.xaxis.set_major_locator(MultipleLocator(1))

ax.yaxis.set_minor_locator(MultipleLocator(0.05))
ax.yaxis.set_major_locator(MultipleLocator(0.1))


set_grid(ax)
plt.tight_layout()


plt.savefig(main_folder + "Figures_chap_article/figures_pour_elsa/figure_3_c_only.png",dpi=dpi_global)
plt.savefig(main_folder + "Figures_chap_article/figures_pour_elsa/figure_3_c_only.pdf")
plt.savefig(main_folder + "Figures_chap_article/figures_pour_elsa/figure_3_c_only.svg")
plt.close('all')













fig,ax=plt.subplots(1)
fig.set_size_inches((90*mm,70*mm))


nice_plot(new_data["creep_center"]/100-new_data["creep_sides"]/100,new_data["mean_freq"],ax=ax,
            xerr=new_data["sigma_creep_center"]/100,yerr=new_data["wide_freq"],
            xlabel=r"$S^{eye}_f-S^{solid}_f$" ,ylabel=freq_name_short ,
            old_data=old_data3,in_solid=in_solid,
            color=main_plot_color, error_bar_color=error_bar_color,
            marker_size=scatter_size, capsize=error_bar_width, marker="d")





# right
ax.set_xlim([-2/100,18/100])
ax.xaxis.set_minor_locator(MultipleLocator(0.025))
ax.xaxis.set_major_locator(MultipleLocator(0.05))

ax.set_ylim([-0.01,0.6])
ax.yaxis.set_minor_locator(MultipleLocator(0.25/2))
ax.yaxis.set_major_locator(MultipleLocator(0.25))


set_grid(ax)
plt.tight_layout()


plt.savefig(main_folder + "Figures_chap_article/figures_pour_elsa/figure_3_inset_only.png",dpi=dpi_global)
plt.savefig(main_folder + "Figures_chap_article/figures_pour_elsa/figure_3_inset_only.pdf")
plt.savefig(main_folder + "Figures_chap_article/figures_pour_elsa/figure_3_inset_only.svg")
plt.close('all')
























