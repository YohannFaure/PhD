## Imports
# Science
# Location
main_folder = 'E:/manuscrit_these/'
main_folder = '../'


with open(main_folder + "parameters.py", 'r') as f:
        exec(f.read())



loaded_data = np.load(main_folder + "data_for_figures/figure_2.npy",allow_pickle=True).all()
locals().update(loaded_data)


###



def sliding_plot(x,disp,ax,vlines = None,labels=True,suptitle=False,color="blue",alpha=1):

    if not vlines is None:
        for v in vlines:
            #ax.axvline(v,color="black",linestyle=dash_line_style ,linewidth=0.4,alpha= secondary_plot_alpha)
            ax.plot([v,v],[0,disp[np.abs(x - v).argmin()]],
                    color=dashed_color,linestyle=dash_line_style,
                    linewidth=0.1,alpha= 0.2)

    ax.grid(True,which="both")

    ax.plot(x,disp,color=color,alpha=alpha)


    if labels:
        ax.set_ylabel(tot_sliding_short )

    if suptitle:
        ax.set_title(suptitle)


def creep_plot(x,creep,ax,labels=True,suptitle=False,color="blue",alpha=1,marker=scatter_marker,s=scatter_size,vlines=False,label=None ):
    if vlines:
        for i in range(len(x)):
            ax.plot([x[i],x[i]],[creep[i],100],
                    color=dashed_color,linestyle=dash_line_style,
                    linewidth=0.1,alpha=0.2)

    ax.plot(x,creep,color=color,alpha=alpha,label=label)
    ax.scatter(x,creep,marker=marker ,s=s ,c=color,alpha = 1 ,zorder=10,facecolors="k",linewidth=0.01)

    ax.grid(True,which="both")

    if labels:
        ax.set_ylabel(sliding_perc_name_short )

    if suptitle:
        ax.set_title(suptitle)




def mu_plot(t,mu,ax,timings=None,linecolor=main_plot_color):
    a=next((i for i, b in enumerate(mu[7000:]>0.05) if b), None)+7000
    start = max(a-5200,0)
    ax.plot(t[start:]-t[start],mu[start:],color=linecolor )



##

time_lim_1=[0,220]
time_lim_2=[0,180]



fig, axes = plt.subplots(nrows=2,ncols=2,sharex="col",sharey="none")
fig.set_size_inches(size_fig_article_2)


colors_here=[solid_solid_color,main_plot_color]


# main figure
colors_sides=[solid_in_granular_color,solid_in_granular_color]
colors_center=[hole_in_solid_color,granular_color]
markers=[solid_marker,scatter_marker]
markers_size=[scatter_size*2/3,scatter_size]
for i in range(2):
    sliding_plot(x1s[i],disp_sides[i],axes[0,i],vlines = None,labels=True,suptitle=False,
                    color=colors_sides[i],alpha=1)
    sliding_plot(x1s[i],disp_centers[i],axes[0,i],
                    vlines = x2s[i][1:],labels=True,suptitle=False,
                    color=colors_center[i])
    creep_plot(x2s[i],creep_percent_sides[i]/100,
               axes[1,i],labels=True,suptitle=False,
               color=colors_sides[i],alpha = 1,#.2
               vlines=True, marker = markers[i],s=markers_size[i])
    creep_plot(x2s[i],creep_percent_centers[i]/100,axes[1,i],labels=True,suptitle=False,
                color=colors_center[i],vlines=False, marker = markers[i],s=markers_size[i])




# adjust ticks
# top left
axes[0,0].set_xlim(time_lim_1)
axes[0,0].set_ylim([0,3.9])
axes[0,0].set_yticks([0,1,2,3])
axes[0,0].xaxis.set_minor_locator(MultipleLocator(50))
axes[0,0].xaxis.set_major_locator(MultipleLocator(100))

# top right
axes[0,1].set_xlim(time_lim_2)
axes[0,1].set_ylim([0,2.60])
axes[0,1].set_yticks([0,1,2])
axes[0,1].set_yticklabels([0,1,2])
axes[0,1].xaxis.set_minor_locator(MultipleLocator(50))
axes[0,1].xaxis.set_major_locator(MultipleLocator(100))

# bottom left
axes[1,0].set_ylim([0,2.9/100])
axes[1,0].set_yticks([0,1/100,2/100])
axes[1,0].set_yticklabels(["0","0.01","0.02"])
# axes[1,0].set_yticks([0,1/100,2/100,3/100])
# axes[1,0].set_yticklabels(["0","0.01","0.02","0.03"])
#axes[1,0].text(3, 3.47/100, r"$\times 10^{-2}$",size=SMALL_SIZE)
axes[1,0].set_xlabel("temps (s)")

# bottom right
axes[1,1].set_ylim([0,38/100])
axes[1,1].set_yticks([0,10/100,20/100,30/100])
axes[1,1].set_yticklabels(["0","0.1","0.2","0.3"])
#axes[1,1].text(3, 3.47/10, r"$\times 10^{-2}$",size=SMALL_SIZE)
axes[1,1].set_xlabel("temps (s)")



# adjust spacing
real_tight_layout(fig)
plt.subplots_adjust(wspace=0.22,hspace=0)

axes[1,1].get_xaxis().labelpad=2
axes[1,0].get_xaxis().labelpad=2

axes[0,0].yaxis.set_label_coords(-0.23, 0.5)
axes[1,0].yaxis.set_label_coords(-0.24, 0.5)

axes[0,1].yaxis.set_label_coords(-0.18, 0.5)
axes[1,1].yaxis.set_label_coords(-0.19, 0.5)





# adding the insets


# These are in unitless percentages of the figure size. (0,0 is bottom left)
coords = [#left, bottom, width, height
         [0.24,   0.84,    0.14,  0.13],
         [0.727,     0.84,    0.14,  0.13],
         [0.343,    0.375,    0.15,  0.14],
         [0.83,   0.375,    0.15,  0.14],
         ]

# create the axes :
axesin=np.array([fig.add_axes(coords[i]) for i in range(4)])

legends = [["œil (vide)" , "solide"],
           ["œil (rempli)", "solide"]]

bbox = [(0.8,0.47),
        (0.9,0.47)]

# Bottom
for i in range(2):
    axin=axesin[2+i]

    creep_plot(x2s[i],
               (creep_dist_lefts[i]+creep_dist_rights[i])/2,
               axin,labels=None,suptitle=False,
               color=colors_sides[i],
               alpha = 1,# secondary_plot_alpha,
               marker="|",
               s=scatter_size/4,
               label = legends[i][1])
    creep_plot(x2s[i],
               creep_dist_centers[i],axin,labels=None,suptitle=False,
               color=colors_center[i],alpha = 1,marker="|",s=scatter_size/4,
               label = legends[i][0])

    if i==0:
        axin.set_xlim(time_lim_1)
    if i==1:
        axin.set_xlim(time_lim_2)
    axin.xaxis.set_minor_locator(MultipleLocator(50))
    axin.xaxis.set_major_locator(MultipleLocator(100))

    axin.get_yaxis().labelpad=inset_label_pad

    axin.set_ylabel(sliding_ie_name_short)
    axin.set_xlabel("temps (s)",labelpad=inset_label_pad)
    if i==0:
        axin.set_ylim([0,0.038])
        axin.set_yticks([0,0.02])
        axin.set_yticklabels([0,0.02])
        axin.yaxis.set_minor_locator(MultipleLocator(0.01))


    else:
        axin.set_ylim([0,0.27])
        axin.set_yticks([0,0.2])
        axin.set_yticklabels([0,0.2])
        axin.yaxis.set_minor_locator(MultipleLocator(0.1))
    if i==0:
        axin.xaxis.set_label_coords(0.5, -0.3)
        axin.yaxis.set_label_coords(-0.42, 0.5)
    if i==1:
        axin.xaxis.set_label_coords(0.5, -0.3)
        axin.yaxis.set_label_coords(-0.32, 0.5)

    set_up_inset(axin)
    #get handles and labels
    handles, labels = axin.get_legend_handles_labels()
    order = [1,0]
    axin.legend([handles[idx] for idx in order],[labels[idx] for idx in order],prop={'size': 6},framealpha = 0, facecolor = (1,1,1,0),handlelength =.5, loc='lower right',bbox_to_anchor=bbox[i],handletextpad=0.5)




#top
for i in range(2):
    axin=axesin[i]


    mu_plot(times[i],mus[i],axin,timings=timings[i],linecolor=colors_here[i])


    if i==0:
        axin.set_xlim(time_lim_1)
    if i==1:
        axin.set_xlim(time_lim_2)
    axin.xaxis.set_minor_locator(MultipleLocator(50))
    axin.xaxis.set_major_locator(MultipleLocator(100))
    axin.set_ylim([0,0.365])
    axin.set_yticks([0,0.15,0.3])
    axin.set_yticklabels([0,0.15,0.3])




    axin.set_ylabel(r"$\mu$",labelpad=inset_label_pad)
    axin.set_xlabel("temps (s)",labelpad=inset_label_pad)



    #axin.yaxis.tick_right()
    #axin.yaxis.set_label_position("right")


    set_up_inset(axin)



# adding the dashed lines indicating the final value of S
x_top = 133
x_bot = 146


x_dash   = x2s[-1][-5:]

y_dash_1 = np.ones_like(x_dash)*np.median(creep_percent_centers[-1][-5:])/100

y_dash_2 = np.ones_like(x_dash)*np.median(creep_percent_sides[-1][-5:])/100

axes[1,1].plot(x_dash,y_dash_1,linestyle = dash_line_style, c=dashed_color,zorder=10)
axes[1,1].plot(x_dash,y_dash_2,linestyle = dash_line_style, c=dashed_color,zorder=10)

axes[1,1].arrow(x_top, 0, 0, y_dash_1[-1],length_includes_head=True,linewidth=.5,head_width=3,head_length=y_dash_1[-1]/10,color=granular_color,zorder=10)
axes[1,1].arrow(x_bot, 0, 0, y_dash_2[-1],length_includes_head=True,linewidth=.5,head_width=3,head_length=y_dash_1[-1]/10,color=solid_in_granular_color,zorder=10)

axes[1,1].text(x_top-35,y_dash_2[-1]*2.2,r"$S^{eye}_f$",c=granular_color,size=MEDIUM_SIZE)
axes[1,1].text(x_bot-9,y_dash_2[-1]*1.6,r"$S^{solid}_f$",c=solid_in_granular_color,size=MEDIUM_SIZE)


from matplotlib.transforms import TransformedBbox, blended_transform_factory
import matplotlib.patches as patches


set_grid(axes)
for axin in axesin:
    set_grid(axin)
for ax in axesin:
    ax.grid(False,which="both")


fig.subplots_adjust(left=0.15, right=0.99, top=0.98, bottom=0.1,wspace=0.38)


y=(size_fig_1[1]-3*mm)/size_fig_1[1]
fig.text(0.01,y,"a.",size=LETTER_SIZE, weight="bold")
fig.text(.52,y,"b.",size=LETTER_SIZE, weight="bold")
fig.text(0.01,y/1.88,"c.",size=LETTER_SIZE, weight="bold")
fig.text(.52,y/1.88,"d.",size=LETTER_SIZE, weight="bold")


plt.savefig(main_folder + "Figures_chap_article/figure_2.png",dpi=dpi_global)
plt.savefig(main_folder + "Figures_chap_article/figure_2.pdf")
plt.savefig(main_folder + "Figures_chap_article/figure_2.svg")

plt.close('all')



### Figure pour Elsa

fig,ax=plt.subplots(1)
fig.set_size_inches((90*mm,70*mm))



i=0

axin=ax


mu_plot(times[i],mus[i],axin,timings=timings[i],linecolor=colors_here[i])

if i==0:
    axin.set_xlim(time_lim_1)
if i==1:
    axin.set_xlim(time_lim_2)

axin.xaxis.set_minor_locator(MultipleLocator(50))
axin.xaxis.set_major_locator(MultipleLocator(100))

axin.yaxis.set_minor_locator(MultipleLocator(0.15/2))
axin.yaxis.set_major_locator(MultipleLocator(0.15))

axin.set_ylim([0,0.365])
axin.set_yticks([0,0.15,0.3])
axin.set_yticklabels([0,0.15,0.3])


axin.set_ylabel(r"$\mu$")
axin.set_xlabel("time (s)")



plt.tight_layout()


#ax.yaxis.set_minor_locator(MultipleLocator(0.05))
#ax.yaxis.set_major_locator(MultipleLocator(0.1))

set_grid(ax)

plt.savefig(main_folder + "Figures_chap_article/figures_pour_elsa/figure_2_inset_1.png",dpi=dpi_global)
plt.savefig(main_folder + "Figures_chap_article/figures_pour_elsa/figure_2_inset_1.pdf")
plt.savefig(main_folder + "Figures_chap_article/figures_pour_elsa/figure_2_inset_1.svg")
plt.close('all')



fig,ax=plt.subplots(1)
fig.set_size_inches((90*mm,70*mm))



i=0

axin=ax


creep_plot(x2s[i],
            (creep_dist_lefts[i]+creep_dist_rights[i])/2,
            axin,labels=None,suptitle=False,
            color=colors_sides[i],
            alpha = 1,# secondary_plot_alpha,
            marker="o",
            s=scatter_size,
            label = "solid")
creep_plot(x2s[i],
            creep_dist_centers[i],axin,labels=None,suptitle=False,
            color=colors_center[i],alpha = 1,marker="o",s=scatter_size,
            label = "hole (empty)")

if i==0:
    axin.set_xlim(time_lim_1)
if i==1:
    axin.set_xlim(time_lim_2)

axin.xaxis.set_minor_locator(MultipleLocator(50))
axin.xaxis.set_major_locator(MultipleLocator(100))


axin.set_ylabel(sliding_ie_name_short)
axin.set_xlabel("time (s)")

if i==0:
    axin.set_ylim([0,0.038])
    axin.set_yticks([0,0.02])
    axin.set_yticklabels([0,0.02])
    axin.yaxis.set_minor_locator(MultipleLocator(0.01))

else:
    axin.set_ylim([0,0.27])
    axin.set_yticks([0,0.2])
    axin.set_yticklabels([0,0.2])
    axin.yaxis.set_minor_locator(MultipleLocator(0.1))


set_grid(ax)


#get handles and labels
handles, labels = axin.get_legend_handles_labels()
order = [1,0]
axin.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
plt.tight_layout()







plt.savefig(main_folder + "Figures_chap_article/figures_pour_elsa/figure_2_inset_3.png",dpi=dpi_global)
plt.savefig(main_folder + "Figures_chap_article/figures_pour_elsa/figure_2_inset_3.pdf")
plt.savefig(main_folder + "Figures_chap_article/figures_pour_elsa/figure_2_inset_3.svg")
plt.close('all')









fig,ax=plt.subplots(1)
fig.set_size_inches((90*mm,70*mm))



i=0

creep_plot(x2s[i],creep_percent_sides[i]/100,
                ax,
                labels=True,
                suptitle=False,
                color=colors_sides[i],
                alpha = 1,
                vlines=False,
                marker = markers[i],
                s=scatter_size)
creep_plot(x2s[i],creep_percent_centers[i]/100,
                ax,
                labels=True,suptitle=False,
                color=colors_center[i],
                vlines=False,
                marker = markers[i],
                s=scatter_size)




ax.set_xlim(time_lim_1)
ax.xaxis.set_minor_locator(MultipleLocator(50))
ax.xaxis.set_major_locator(MultipleLocator(100))
ax.yaxis.set_minor_locator(MultipleLocator(0.0025))
ax.yaxis.set_major_locator(MultipleLocator(0.005))


# bottom left
ax.set_ylim([0,1/100])
ax.set_xlabel("time (s)")

plt.tight_layout()

set_grid(ax)



plt.savefig(main_folder + "Figures_chap_article/figures_pour_elsa/figure_2_c_only.png",dpi=dpi_global)
plt.savefig(main_folder + "Figures_chap_article/figures_pour_elsa/figure_2_c_only.pdf")
plt.savefig(main_folder + "Figures_chap_article/figures_pour_elsa/figure_2_c_only.svg")
plt.close('all')






















fig,ax=plt.subplots(1)
fig.set_size_inches((90*mm,70*mm))



i=0

sliding_plot(x1s[i],disp_sides[i],ax,vlines = None,labels=True,suptitle=False,
                color=colors_sides[i],alpha=1)
sliding_plot(x1s[i],disp_centers[i],ax,
                vlines = x2s[i][1:],labels=True,suptitle=False,
                color=colors_center[i],alpha=1)


ax.set_xlim(time_lim_1)
ax.xaxis.set_minor_locator(MultipleLocator(50))
ax.xaxis.set_major_locator(MultipleLocator(100))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(1))


ax.set_ylim([0,3.9])


ax.set_xlabel("time (s)")

plt.tight_layout()

set_grid(ax)



plt.savefig(main_folder + "Figures_chap_article/figures_pour_elsa/figure_2_a_only.png",dpi=dpi_global)
plt.savefig(main_folder + "Figures_chap_article/figures_pour_elsa/figure_2_a_only.pdf")
plt.savefig(main_folder + "Figures_chap_article/figures_pour_elsa/figure_2_a_only.svg")
plt.close('all')

































