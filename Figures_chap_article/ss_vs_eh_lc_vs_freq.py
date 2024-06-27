### load data : set up the loading

main_folder = 'E:/manuscrit_these/'
main_folder = '../'


with open(main_folder + "parameters.py", 'r') as f:
        exec(f.read())


def mu_plot(t,mu,ax,fn,labels=True,suptitle=False,timings=None,
                      linecolor=main_plot_color,nsmo=1,label=None):
    """
    Plots Mu(t) (top)
    """
    # Create new time and new starting point, to keep only loading
    a=next((i for i, b in enumerate(mu[7000:]>0.05) if b), None)+7000
    start = max(a-5200,0)
    new_t = t[start:]-t[start]
    new_mu = mu[start:]-mu[start]

    # plot
    ax.plot(smooth(new_t,nsmo),smooth(new_mu,nsmo),color=linecolor,label=label )

    if labels:
        ax.set_ylabel("$\mu=F_S\,/\,F_N$",size=MEDIUM_SIZE)

    if suptitle:
        ax.set_title(suptitle)

    # Les petits pointillés

    if timings:
        for ti in timings:
            ax.axvline(ti,linestyle=dash_line_style )
            ax.plot([ti,ti],[0,disp[np.abs(t - ti).argmin()]],
                    color=dashed_color,linestyle=dash_line_style ,
                    linewidth=0.4,alpha= secondary_plot_alpha)



## quick-load data if already generated

dicto = np.load(main_folder + "data_for_figures/figure_1_c_d.npy",allow_pickle=True).all()


timings=dicto["timings"]
times=dicto["times"]
mus=dicto["mus"]
mean_fns=dicto["mean_fns"]
load_profiles=dicto["load_profiles"]
LCs=dicto["LCs"]

sm2 = np.load(main_folder + "data_for_figures/slowmon.npy")
time_sm_2 = np.load(main_folder + "data_for_figures/slowmon_time.npy")
time_sm_2=time_sm_2[6000:]
fn2 = -500/3*sm2[15,6000:]-100
fs2 = -500/3*sm2[31,6000:]
fs2-=np.min(fs2)
mean_fn2=np.mean(fn2)
mu2=fs2/fn2













widths = [1.4,1]



fig = plt.figure()

gs = fig.add_gridspec(nrows=1, ncols=2, left=0.12, right=0.99,
                      top=0.95,bottom=0.2,
                      wspace=0.3,width_ratios=widths)

axes = [fig.add_subplot(gs[0]),fig.add_subplot(gs[1])]

ax=axes[0]
fig.set_size_inches(size_fig_ss_vs_eh)


mu_plot(times[0],mus[0],ax,mean_fns[0],labels="à vide",suptitle=False,
                      linecolor=solid_in_granular_color,nsmo=20,label="œil à vide" )

mu_plot(time_sm_2,mu2,ax,mean_fn2,labels="à vide",suptitle=False,
                      linecolor="k" ,nsmo=20, label="interface homogène")

ax.set_xlabel("temps (s)",size=MEDIUM_SIZE,labelpad=2)
set_grid(ax)

ax.set_xlim(0,300)
ax.set_ylim(0,0.4)
ax.xaxis.set_minor_locator(MultipleLocator(50))
ax.xaxis.set_major_locator(MultipleLocator(100))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
ax.yaxis.set_major_locator(MultipleLocator(0.1))






















loaded_data = np.load(main_folder + "data_for_figures/figure_1_b.npy",allow_pickle=True).all()
locals().update(loaded_data)


in_solid=np.isin(datasolid["manip_num"],solids)




###


def nice_plot(xpos,ypos,ax=None,xerr=None,yerr=None,xlabel="",ylabel="",
              annotate=None,old_data=False,
              color="b",error_bar_color="k",marker_size=3,capsize=3,
              marker="d",in_solid=None,solid_color=solid_in_granular_color,cmap=None,label=None):

    # all data
    if not cmap is None:
        ax.errorbar(xpos,ypos, xerr=xerr,yerr=yerr,
                fmt=" ",capsize=capsize,color="k",
                ecolor=error_bar_color,elinewidth=.5,
                alpha=error_bar_alpha,markeredgewidth=markeredgewidth)
        aaaa=ax.scatter(xpos[np.logical_not(in_solid)],ypos[np.logical_not(in_solid)],c=color[np.logical_not(in_solid)],s=marker_size,marker=marker,zorder=5,edgecolors="k",linewidth=0.01,cmap=cmap,vmin=min(color), vmax=1.01*max(color),label=label)


    else:
        ax.errorbar(xpos,ypos, xerr=xerr,yerr=yerr,
                fmt=" ",capsize=capsize,color="k",
                ecolor=error_bar_color,
                elinewidth=.5,alpha=error_bar_alpha,markeredgewidth=markeredgewidth)
        ax.scatter(xpos,ypos,color=color,s=marker_size,marker=marker,zorder=5,edgecolors="k",linewidth=0.01,label=label)
    # solids
    if not in_solid is None:
        ax.errorbar(xpos[in_solid],ypos[in_solid], xerr=xerr[...,in_solid],yerr=yerr[...,in_solid],
                    fmt=" ",capsize=capsize,color=solid_color,ecolor=error_bar_color,elinewidth=.5,alpha=error_bar_alpha ,markeredgewidth=markeredgewidth)
        ax.scatter(xpos[in_solid],ypos[in_solid],color=solid_color,s=marker_size*2/3,marker=solid_marker,zorder=5,edgecolors="k",linewidth=0.01)

    ax.set_xlabel(xlabel,size=MEDIUM_SIZE,labelpad=2)
    ax.set_ylabel(ylabel,size=MEDIUM_SIZE,labelpad=2)
    if not cmap is None:
        return(aaaa)

###

ax2=axes[1]


nice_plot(datagran['mean_eps_yy_ss']/1e6  * 1200  ,datagran['mean_dt'],ax=ax2,
        xerr=datagran['std_eps_yy_ss']*0,yerr=datagran['sigma_dt'],
        xlabel=r"$\left<\sigma_{yy}^{solid}\right>$ (MPa)",ylabel=r"$\left<\Delta T\right>$ (s)",
        in_solid=None,
        color=solid_in_granular_color, error_bar_color=error_bar_color,
        marker_size=scatter_size*2, capsize=error_bar_width, marker="o",label="œil à vide" )


aaaa=nice_plot(datasolid['mean_eps_yy_ss'][in_solid]/1e6   * 1200   ,datasolid['mean_dt'][in_solid],ax=ax2,
        xerr=datasolid['std_eps_yy_ss'][in_solid]*0,yerr=datasolid['sigma_dt'][in_solid],
        xlabel=r"$\left<\sigma_{yy}^{solid}\right>$ (MPa)",ylabel=r"$\left<\Delta T\right>$ (s)",
        in_solid=None,
        color=solid_in_granular_color, error_bar_color=error_bar_color,
        marker_size=scatter_size*2, capsize=error_bar_width, marker="o")


nice_plot(ss_sig/1e6  *1200  ,ss_times_mean,ax=ax2,
        xerr=ss_sig*0,yerr=ss_times_std,
        xlabel=r"$\left<F_N\right>$ (N)",ylabel=r"$\left<\Delta T\right>$ (s)",
        in_solid=None,
        color='k', error_bar_color=error_bar_color,
        marker_size=scatter_size*2, capsize=error_bar_width, marker='s',
        solid_color="k",label="interface homogène")



ax2.xaxis.set_minor_locator(MultipleLocator(750))
ax2.xaxis.set_major_locator(MultipleLocator(1500))
ax2.yaxis.set_minor_locator(MultipleLocator(10))
ax2.yaxis.set_major_locator(MultipleLocator(20))

#ax2.tick_params(labelsize=SMALL_SIZE)

#plt.plot([1,3.2],[14,62],color="k",linewidth=.5,linestyle=":")

ax2.set_xlim([0,4300])
ax2.set_ylim([0,65])

set_grid(ax2)






ax.legend(facecolor = (1,1,1,1),framealpha=1)



fig.text(0.02,0.94,"a.",size=LETTER_SIZE, weight='bold')
fig.text(0.58,0.94,"b.",size=LETTER_SIZE, weight='bold')










plt.savefig(main_folder + "Figures_chap_article/ss_vs_eh.svg")
plt.savefig(main_folder + "Figures_chap_article/ss_vs_eh.pdf")
plt.savefig(main_folder + "Figures_chap_article/ss_vs_eh.png",dpi=dpi_global)
#plt.show()
plt.close('all')


###

fig,ax=plt.subplots(1)
fig.set_size_inches(size_fig_lc_freq)

nice_plot(datasolid['mean_lc'],datasolid['mean_freq'],ax=ax,
        xerr=datasolid['wide_lc'],yerr=datasolid['wide_freq'],
        xlabel=r"$C_\sigma$",ylabel=r"$\left<1/\Delta T\right>$ (Hz)",
        in_solid=in_solid,
        color=solid_in_granular_color, error_bar_color=error_bar_color,
        marker_size=scatter_size*3, capsize=error_bar_width, marker="d",solid_color=solid_solid_color)


ax.xaxis.set_minor_locator(MultipleLocator(.5))
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
ax.yaxis.set_major_locator(MultipleLocator(0.1))


set_grid(ax)
plt.tight_layout()

fig.subplots_adjust(left=0.2, right=0.99, bottom=0.16, top=0.99)

plt.savefig(main_folder + "Figures_chap_article/lc_vs_freq.svg")
plt.savefig(main_folder + "Figures_chap_article/lc_vs_freq.pdf")
plt.savefig(main_folder + "Figures_chap_article/lc_vs_freq.png",dpi=dpi_global)
plt.close('all')






