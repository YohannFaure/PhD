### load data : set up the loading
main_folder = 'E:/manuscrit_these/'
main_folder = '../'


with open(main_folder + "parameters.py", 'r') as f:
        exec(f.read())


def load_profile_plot(x_plot,load_profile,ax,labels=True,suptitle=False,legend=False,
                      linecolor=main_plot_color,c_solid=solid_in_granular_color,c_hole=granular_color, marker=scatter_marker,marker_size=3 ):
    """
    To plot the loading profile (bottom)
    """
    #load
    load_profile_mean=np.mean(load_profile,axis=0)

    # Les lignes transparentes
    for event in load_profile:
        ax.plot(x_plot,event*1e-6,c=secondary_plot_color ,alpha=secondary_plot_alpha ,zorder=1)
        ax.plot(x_plot,load_profile_mean*1e-6,color=linecolor ,zorder=2)

    # Le plot en lui même
    # Points bords
    ax.scatter(x_plot,load_profile_mean*1e-6,
               marker=marker ,s=scatter_size*marker_size ,c=c_solid,zorder=3,edgecolors="k",linewidth=0.01 )
    # Point central

    ax.scatter(x_plot[5],load_profile_mean[5]*1e-6,
               marker=marker,s=scatter_size*marker_size,
               c=c_hole ,label=legend,zorder=3,edgecolors="k",linewidth=0.01)
    if legend:
        ax.legend()

    if labels:
        ax.set_ylabel("$\sigma_{yy}$ (MPa)",size=MEDIUM_SIZE)

    if suptitle:
        ax.set_title(suptitle)


def mu_plot(t,mu,ax,fn,labels=True,suptitle=False,timings=None,
                      linecolor=main_plot_color):
    """
    Plots Mu(t) (top)
    """
    # Create new time and new starting point, to keep only loading
    a=next((i for i, b in enumerate(mu[7000:]>0.05) if b), None)+7000
    start = max(a-5200,0)
    new_t = t[start:]-t[start]
    new_mu = mu[start:]

    # plot
    ax.plot(new_t,new_mu,color=linecolor )

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



def trig_to_time(sm_trig):
    sm_trig=sm_trig>3
    sm_trig[4:-1]+=sm_trig[0:-5]
    sm_trig=sm_trig>0.5
    stop = len(sm_trig)
    j=10000
    timings=[]
    while j < stop:
        if sm_trig[j] :
            k=0
            while j+k<stop and sm_trig[j+k]:
                k+=1
            if k>25:
                timings.append(j)
            j=j+k
        else :
            j+=1
    return(timings)






## quick-load data if already generated

dicto = np.load(main_folder + "data_for_figures/figure_1_c_d.npy",allow_pickle=True).all()


timings=dicto["timings"]
times=dicto["times"]
mus=dicto["mus"]
mean_fns=dicto["mean_fns"]
load_profiles=dicto["load_profiles"]
LCs=dicto["LCs"]



#fig, axes = plt.subplots(nrows=2,ncols=4,sharex=False,sharey="row")
#fig.set_size_inches(size_fig_1)
fig = plt.figure(layout=None)
fig.set_size_inches(size_fig_article_1)


heights = [1,1]
widths = [1,0.1,1,1,1]


gs = fig.add_gridspec(nrows=2, ncols=5, left=0.085, right=0.99,
                      top=0.95,bottom=0.1,
                      hspace=.3, wspace=0,width_ratios=widths,
                      height_ratios=heights)


#top
axes_top=[]
for i in range(5):
    if i!=1:
        axes_top.append( fig.add_subplot(gs[0, i]) )
        axes_top[-1].sharex(axes_top[0])
        axes_top[-1].sharey(axes_top[0])

axes_bot=[]
for i in range(5):
    if i!=1:
        axes_bot.append( fig.add_subplot(gs[1, i]) )
        axes_bot[-1].sharex(axes_bot[0])
        axes_bot[-1].sharey(axes_bot[0])

axes=np.array([axes_bot,axes_top])


# plot solid solid
i=0
load_profile_plot(x_plot, load_profiles[i], axes[1,i],
                      labels=i<1, legend=False,
                      linecolor=solid_solid_color,
                      c_solid=solid_in_granular_color,
                      c_hole=hole_in_solid_color,
                      marker=solid_marker,marker_size=2)
mu_plot(times[i],mus[i],axes[0,i],mean_fns[i],labels=i<1,suptitle=False,
                      linecolor=solid_solid_color )

# plot granular
for i in range(1,4):
    load_profile_plot(x_plot, load_profiles[i], axes[1,i], labels=i<1, legend=False,c_hole=granular_color,linecolor=main_plot_color)
    mu_plot(times[i],mus[i],axes[0,i],mean_fns[i],labels=i<1,suptitle=False,linecolor=main_plot_color)




# top
for i in range(4):
    axes[0,i].set_xlim([0,230])
    if i==0:
        axes[0,i].set_xticks([0,100,200])
    else:
        axes[0,i].set_xticks([100,200])
    axes[0,i].xaxis.set_minor_locator(MultipleLocator(50))
    axes[1,i].set_xlim([0,150])
    axes[1,i].set_xticks([0,50,100])
axes[0,0].set_ylim([0,0.4])
axes[0,0].set_yticks([0,0.1,0.2,0.3])
axes[0,0].set_yticklabels([0,0.1,0.2,0.3])
#axes[0,2].set_xlabel("time (s)",size=MEDIUM_SIZE,labelpad=2)
axes[0,0].set_xlabel("temps (s)",size=MEDIUM_SIZE,labelpad=2)
#axes[0,1].set_ylabel("$\mu=F_S\,/\,F_N$",size=MEDIUM_SIZE,labelpad=7)

# bottom

axes[1,0].set_ylim([0,5.5])
axes[1,0].set_yticks([0,2,4])

# adjust spacings and limits
#real_tight_layout(fig)
set_grid(axes)
plt.subplots_adjust(wspace=0)
plt.subplots_adjust(hspace=0.35)



for i in range(1,4):
    plt.setp(axes[0,i].get_yticklabels(), visible=False)
    plt.setp(axes[1,i].get_yticklabels(), visible=False)


#fig.subplots_adjust(left=0.14,right=0.99, top=0.99, bottom=0.11)


#axes[1,2].set_xlabel("$x$ (mm)",size=MEDIUM_SIZE,labelpad=2)
axes[1,0].set_xlabel("$x$ (mm)",size=MEDIUM_SIZE,labelpad=2)
#axes[1,1].set_ylabel("$\sigma_{yy}$ (MPa)",size=MEDIUM_SIZE,labelpad=5.5)

axes[0,0].yaxis.labelpad=5
axes[1,0].yaxis.labelpad=9
axes[0,1].xaxis.set_label_coords(1.45, -0.18)
axes[1,1].xaxis.set_label_coords(1, -0.18)

for i in range(4):
    axes[1,i].text(5,5.05,r"$\left<F_N\right>={:.0f}$ N".format(round(dicto["mean_fns"][i])*10),size=7)
    axes[1,i].text(145,5.05,r"${}=${:.2f}".format(LC_name_short,(dicto["LCs"][i])),size=7,horizontalalignment='right')

axes[1,0].set_title("œil à vide",size=MEDIUM_SIZE,pad=3)
axes[1,2].set_title("granulaire (œil rempli)",size=MEDIUM_SIZE,pad=3)


fig.text(0.007,0.95,"a.",size=LETTER_SIZE, weight='bold')
fig.text(0.007,0.47,"b.",size=LETTER_SIZE, weight='bold')
# save

plt.savefig(main_folder + "Figures_chap_article/figure_1_c_d.svg")
plt.savefig(main_folder + "Figures_chap_article/figure_1_c_d.pdf")
plt.savefig(main_folder + "Figures_chap_article/figure_1_c_d.png",dpi=dpi_global)
#plt.show()
plt.close('all')
















