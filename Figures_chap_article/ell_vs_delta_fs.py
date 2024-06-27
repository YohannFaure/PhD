
### EN COURS DE BOULOT


### load data : set up the loading

main_folder = 'E:/manuscrit_these/'
main_folder = '../'


with open(main_folder + "parameters.py", 'r') as f:
        exec(f.read())




## data sm
data_sm = np.load(main_folder + "data_for_figures/summary_data.npy",allow_pickle=True).all()
mean_fd = np.array(data_sm["mean_fd"])
lc_sm = data_sm["mean_lc"]

### data \ell

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


whichy = np.cumsum(np.diff(lc_event)!=0)
whichy = np.insert(whichy,0,0)

mean_l = np.array([np.mean(ell[whichy==i]) for i in range(34)])


##

S=1.5e-3




bins = np.array([-1,-0.5,0,1,2])
binned = (bins[1:]+bins[:-1])/2

def make_hists(bins, bin_variable, to_count):
    n_bin=len(bins)-1
    hists = []
    stds = []
    for i in range(n_bin):
        mask = np.logical_and(bin_variable>bins[i],bin_variable<bins[i+1])
        hists.append(np.mean(to_count[mask]))
        stds.append(np.std(to_count[mask]))
    hists=np.array(hists)
    return(np.array([hists,stds]))


delta_sig_binned,delta_sig_binned_stds = make_hists(bins, lc_sm[np.logical_not(is_solid)], mean_fd[np.logical_not(is_solid)]) * 10/ S / 1e6
delta_sig_solid ,delta_sig_solid_stds = np.mean(mean_fd[is_solid])* 10/ S / 1e6 , np.std(mean_fd[is_solid])* 10/ S / 1e6


ell_binned,ell_binned_stds = make_hists(bins, lc_sm[np.logical_not(is_solid)], mean_l[np.logical_not(is_solid)])/1000
ell_solid ,ell_solid_stds = np.mean(mean_l[is_solid])/1000,np.std(mean_l[is_solid])/1000





###

fig,ax_0=plt.subplots()
fig.set_size_inches(size_fig_ell_vs_delta_fs)

ax_0.errorbar(delta_sig_binned,1/np.sqrt(ell_binned-0.03),
                xerr = delta_sig_binned_stds,
                yerr = ell_binned_stds/(2 * (ell_binned-0.03)**(3/2)),
                fmt=" ",capsize=error_bar_width,color="k",
                ecolor=error_bar_color,
                elinewidth=.5,alpha=error_bar_alpha,markeredgewidth=markeredgewidth)

ax_0.scatter(delta_sig_binned,1/np.sqrt(ell_binned-0.03),
        c=main_plot_color,s=scatter_size*2,
        marker="d",zorder=2,
        edgecolors="k",linewidth=0.01)


ax_0.errorbar(delta_sig_solid,1/np.sqrt(ell_solid-0.03),
                xerr = delta_sig_solid_stds,
                yerr = ell_solid_stds/(2 * (ell_solid-0.03)**(3/2)),
                fmt=" ",capsize=error_bar_width,color="k",
                ecolor=error_bar_color,
                elinewidth=.5,alpha=error_bar_alpha,markeredgewidth=markeredgewidth)


ax_0.scatter(delta_sig_solid,1/np.sqrt(ell_solid-0.03),
        c=solid_solid_color,s=scatter_size*4/3,
        marker="o",zorder=2,
        edgecolors="k",linewidth=0.01)




ax_0.scatter(10*mean_fd/S/1e6,
                1/np.sqrt(mean_l/1000-0.03),
                marker = scatter_marker, s = scatter_size/4,
                c=secondary_plot_color,alpha = secondary_plot_alpha ,
                zorder=-1,edgecolors="k",linewidth=0)


ax_0.xaxis.set_minor_locator(MultipleLocator(0.1))
ax_0.xaxis.set_major_locator(MultipleLocator(0.2))
ax_0.yaxis.set_minor_locator(MultipleLocator(1))
ax_0.yaxis.set_major_locator(MultipleLocator(2))


ax_0.set_xlim([0,0.6])
ax_0.set_ylim([0,11])

plt.xlabel("$\Delta\sigma$ macro (MPa)")
plt.ylabel("$\ell$ (m)")
plt.tight_layout()
set_grid(ax_0)

fig.savefig(main_folder + "Figures_chap_article/ell_vs_delta_fs.png",dpi=dpi_global)
fig.savefig(main_folder + "Figures_chap_article/ell_vs_delta_fs.pdf")
fig.savefig(main_folder + "Figures_chap_article/ell_vs_delta_fs.svg")
plt.close('all')











