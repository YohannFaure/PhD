## Imports
# Science
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec


main_folder = 'E:/manuscrit_these/'
main_folder = '../'


with open(main_folder + "parameters.py", 'r') as f:
        exec(f.read())





loaded_data = np.load(main_folder + "data_for_figures/figure_5.npy",allow_pickle=True).all()
locals().update(loaded_data)



########################## create figure



# nested gridspecs, yeaaaaah...
fig = plt.figure()
fig.set_size_inches(size_fig_article_4)

gs0 = gridspec.GridSpec(2, 1, figure=fig, height_ratios = [9,3],
                      left=0.07, right=0.97,
                      top=0.97,bottom=0.07,
                      hspace=0.22, wspace=0)

#nrows=nrows, ncols=11, left=0.06, right=0.97,
#                      top=0.98,bottom=0.06,
#                      hspace=0, wspace=0

gs00 = gridspec.GridSpecFromSubplotSpec(10, 3, subplot_spec=gs0[0],
                      width_ratios = [5,0.8,5],
                      hspace=0, wspace=0)

axes_1=[]
axes_3=[]

for i in range(10):
    axes_1.append( fig.add_subplot(gs00[i,0]) )
    axes_1[-1].sharex(axes_1[0])

    axes_3.append( fig.add_subplot(gs00[i, -1]) )
    axes_3[-1].sharex(axes_3[0])

axes_3,axes_1=axes_1,axes_3


gs01 = gridspec.GridSpecFromSubplotSpec(1, 7, subplot_spec = gs0[1],
                      width_ratios = [1,1,1,1,1,0.6,1.5],
                      hspace=0, wspace=0)
axes_2=[]
for i in range(5):
    axes_2.append( fig.add_subplot(gs01[0, i]) )
    axes_2[-1].sharex(axes_2[0])
    axes_2[-1].sharey(axes_2[0])

gs001 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec = gs01[-1],
                      height_ratios = [1,0.1],
                      hspace=0, wspace=0)

ax_4 = fig.add_subplot(gs001[0, 0])



######################################## Data round




## Plot round

start_r=75
stop_r=115

start=np.sum(time_sm<start_r)
stop=np.sum(time_sm<stop_r)


color=[solid_in_granular_color]*5+[granular_color]+[solid_in_granular_color]*4
labels = [r"$x=5 \pm 0.25$ mm"]+[r"$x={}$".format(i) for i in [int(x_plot[i]) for i in range(1,len(x_plot))]]

for i in range(10):
    axes_1[i].plot(time_sm[start:stop],(eps_xy_sm[i][start:stop])*1e3,color=color[i])
    axes_1[i].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axes_1[i].grid("both")
    y_lim=(np.mean(eps_xy_sm[i][start:stop]))*1e3+np.array([-0.12,0.12])
    axes_1[i].set_ylim(y_lim)
    axes_1[i].yaxis.set_major_locator(MultipleLocator(0.1))
    plt.setp(axes_1[i].get_yticklabels(), visible=False)
    axes_1[i].ticklabel_format(style='plain')
    axes_1[i].annotate(labels[i], xy=(0.01, 0.118), xycoords='axes fraction',
                xytext=(10, 0), textcoords='offset points',
                va="center",size=INSET_SMALL_SIZE)

axes_1[-1].xaxis.set_minor_locator(MultipleLocator(5))
axes_1[-1].xaxis.set_major_locator(MultipleLocator(10))

for i in range(9):
    plt.setp(axes_1[i].get_yticklabels(), visible=False)
    plt.setp(axes_1[i].get_xticklabels(), visible=False)


axes_1[1].text(74,0.08,r"$|$",fontsize=10,weight='bold')
axes_1[1].text(72.5,-0.06,r"100 µm/m",rotation=90,size=INSET_SMALL_SIZE)

pos=[(84,0.02),(92,0.02),(96.5,0.02),(100.5,0.02),(102.5,0.02),(106.5,0.02),(111,0.02)]
tipo=['SL','L','L','L','L','L','L']
for i in range(len(pos)):
    axes_1[-1].text(pos[i][0],pos[i][1],tipo[i],size=INSET_SMALL_SIZE)

tipo=['SL','SL','SL','SL','SL','SL','L']
for i in range(len(pos)):
    axes_1[-2].text(pos[i][0],pos[i][1]-0.04,tipo[i],size=INSET_SMALL_SIZE)


event_times = [89,94.2,98.9,101.95,104.4,108.8]
event_indexes = [find_nearest(time_sm,tyy) for tyy in event_times]



for i in range(len(event_times)//2):

    length = event_times[2*i+1]-event_times[2*i]

    if i==0:
        indexes_i = np.array([event_indexes[2*i+1]-800,event_indexes[2*i+1]-200])
    elif i==1:
        indexes_i = np.array([event_indexes[2*i+1]-200,event_indexes[2*i+1]-50])
    else:
        indexes_i = np.array([event_indexes[2*i+1]-300,event_indexes[2*i+1]-100])

    indexes_i_extrap = np.array([event_indexes[2*i]-150,event_indexes[2*i+1]+150])

    x=time_sm[indexes_i]



    y1= np.array([np.mean(eps_xy_sm[-1][indexes_i[0]-30:indexes_i[0]+30]*1e3),
                    np.mean(eps_xy_sm[-1][indexes_i[1]-30:indexes_i[1]+30]*1e3)])
    y2= np.array([np.mean(eps_xy_sm[-2][indexes_i[0]-30:indexes_i[0]+30]*1e3),
                  np.mean(eps_xy_sm[-2][indexes_i[1]-30:indexes_i[1]+30]*1e3)])

    x_extrap = time_sm[indexes_i_extrap]


    def extrap(x,y,x_extrap):
        y_extrap = y[0]+np.diff(y)*(x_extrap-x[0])/np.diff(x)
        return(y_extrap)

    y1_extrap = extrap(x,y1,x_extrap)
    y2_extrap = extrap(x,y2,x_extrap)

    axes_1[-1].plot(x_extrap,y1_extrap,color="k",linestyle = "--")
    axes_1[-2].plot(x_extrap,y2_extrap,color="k",linestyle = "--")

    #axes_1[-1].axvline(x[0])
    #axes_1[-1].axvline(x[1])

# line slipatch
# xs=[0.484]*2
# ys=[0.417,0.853]
# line = mpl.lines.Line2D(xs, ys, lw=1, color='k', alpha=1)
# fig.add_artist(line)

#axes_1[8].text(115.5,0.55,r"Slipping patch",rotation=-90,size=5)



axes_1[-1].set_xlim([start_r,stop_r])
axes_1[-1].set_xlabel('temps (s)')
axes_1[4].set_ylabel(r'$\varepsilon_{xy}$',size=MEDIUM_SIZE)

# js=[7,9]
# ti=np.array([104.8,108])
# ys=[
# [-0.07,0.008],
# [0.07,0.14]
# ]
#
# ys=np.array(ys)
# ys=ys+0.01
#
# for i in range(len(js)):
#     axes_1[js[i]].plot(ti,ys[i],c="k",linewidth=1.2)

mean_lc=data_loaded_1["mean_lc"]
manip_num=np.array(data_loaded_1["manip_num"])
LC_there = mean_lc[manip_num==chosen_manip_fig42][0]


axes_1[0].annotate("${}=${:.2f}".format(LC_name_short,LC_there), xy=(0.75, 0.8), xycoords='axes fraction',
                xytext=(10, 0), textcoords='offset points',
                va="center",size=INSET_SMALL_SIZE)



set_grid(axes_1)


########################## Data for starting point

## Nice plot
def temp_nice_plot(y_min,x, ax = None ,ylim=None) :
    """
    plots a nice representation of the bloc in an histogram
    Useful way later
    """
    if ax is None:
        ax=plt.gca()
    if ylim is None:
        ylim=ax.get_ylim()
    else:
        ylim=ylim
    dilat = (ylim[1]-ylim[0])*0.7
    ax.set_ylim((y_min-0.1*dilat,ylim[1]))
    import matplotlib.patches as patches
    arc_radius = 0.05*dilat
    arc_center_x = 75
    arc_center_y = y_min
    start_angle = 0
    end_angle = 180
    arc_patch = patches.Arc((arc_center_x, arc_center_y), width=30, height=2*arc_radius, angle=0,
                            theta1=start_angle, theta2=end_angle, edgecolor=secondary_plot_color, linewidth=1)
    ax.add_patch(arc_patch)

    # lines of the block
    line1 = patches.ConnectionPatch((0, y_min), (60, y_min), "data", "data", edgecolor=secondary_plot_color, linewidth=1, arrowstyle="-")
    ax.add_patch(line1)
    line2 = patches.ConnectionPatch((90, y_min), (150, y_min), "data", "data", edgecolor=secondary_plot_color, linewidth=1, arrowstyle="-")
    ax.add_patch(line2)
    line3 = patches.ConnectionPatch((0, y_min), (0, y_min+0.25*dilat), "data", "data", edgecolor=secondary_plot_color, linewidth=1, arrowstyle="-")
    ax.add_patch(line3)
    line4 = patches.ConnectionPatch((150, y_min), (150, y_min+0.25*dilat), "data", "data", edgecolor=secondary_plot_color, linewidth=1, arrowstyle="-")
    ax.add_patch(line4)

    # gages, used and unused
    width_patch=(x[1]-x[0])/50
    for xi in x:
        if xi != 75.:
            square_patch = patches.Rectangle((xi-width_patch, y_min+0.07*dilat), 2*width_patch, 0.01*dilat, color=solid_in_granular_color,alpha=1)
        else:
            square_patch = patches.Rectangle((xi-width_patch, y_min+0.07*dilat), 2*width_patch, 0.01*dilat, color=granular_color,alpha=1)
        ax.add_patch(square_patch)


###################################### Histogram round

solids = [14,15,16,17,18,37,38]


data_nuc = np.copy(data_nuc_raw)
mean_lc=data_loaded_2["mean_lc"]
manip_num=data_loaded_2["manip_num"]
is_solid=[i in solids for i in manip_num]


# extract is solid and events
from_solid=data_nuc[0].astype(bool)
from_granular = np.logical_not(from_solid)
n_events = data_nuc[1]
data_nuc=data_nuc[2:]

#reverse gages
data_nuc=(data_nuc[::-1] % 2 ==1).transpose()

# create LC
lc_event=mean_lc[np.cumsum(np.diff(np.insert(n_events,0,0))!=1)]





## Define histograms

n_bin=len(bins)-1


def make_hists_bis(bins, bin_variable, to_count):
    hists = []
    for i in range(n_bin):
        in_bin = np.logical_and(bin_variable>bins[i],bin_variable<bins[i+1])
        hists.append(to_count[in_bin].sum(axis=0)/to_count[in_bin].shape[0])
    hists=np.array(hists)
    return(hists)


histogram_round_solid = data_nuc[from_solid].sum(axis=0)/data_nuc[from_solid].shape[0]
histograms_round=make_hists_bis(bins, lc_event[from_granular], data_nuc[from_granular])



data_nuc[:,5]=True

data_nuc_reel = np.copy(data_nuc)
ell = []
for i in range(len(data_nuc)):
    try:
        slipping = x_plot[data_nuc[i]]
        width = slipping[-1]-slipping[0]
        if slipping[0] > 105. or slipping[-1]<45.:
            width += 45
        if slipping[0] == 75. and slipping[-1] == 75. :
            width = 30
        elif slipping[0] == 75.:
            width += 15
        elif slipping[-1] == 75. :
            width += 15

        ell.append(width)
    except:
        ell.append(30.)

ell=np.array(ell)
#ell[ell==0]=30

# plt.scatter(lc_event, temp)
# plt.xlabel("LC")
# plt.ylabel("width of slipping patch")
# plt.show()


## Plot histogram round

hist_y_lim=[-0.08,1.18]

y_min=-0.05
pad_title = pad_title_hists
y_title = 1+1e-50
width=0.8*min(np.diff(x_plot))

meanstarts_left=[]
meanstarts_right=[]

for i in range(5):
    ax=axes_2[i]

    if i==0:
        ax.bar(x_plot, histogram_round_solid,
                width=width,
                color = solid_solid_color,linewidth=.1,edgecolor='k')
        ax.set_title("à vide", pad=pad_title,y=y_title,size=INSET_MEDIUM_SIZE)
        temp_nice_plot(y_min,x_plot,ax=ax,ylim=(0.1,1.1))
    else:
        ax.bar(x_plot, histograms_round[i-1],
                width=width,
                color = main_plot_color,linewidth=.1,edgecolor='k')
        ax.set_title(r"{}$\leq${}<{}".format(bins[i-1],r"${}$".format(LC_name_short),bins[i]), pad=pad_title, y=y_title,size=INSET_MEDIUM_SIZE)
        temp_nice_plot(y_min,x_plot,ax=ax,ylim=(0.1,1.1))

        meanstart_left = (x_plot*histograms_round[i-1])[x_plot>75.].sum()/histograms_round[i-1][x_plot>75.].sum()
        meanstart_right= (x_plot*histograms_round[i-1])[x_plot<75.].sum()/histograms_round[i-1][x_plot<75].sum()
        #ax.axvline(x=meanstart_left, ymax=0.8,linestyle='--',c='k')
        #ax.axvline(x=meanstart_right, ymax=0.8,linestyle='--',c='k')
        meanstarts_left.append(meanstart_left)
        meanstarts_right.append(meanstart_right)




# adjust ticks
# top
axes_2[0].set_xlim([-5,155])
axes_2[0].set_ylim(hist_y_lim)
axes_2[0].set_yticks([0,0.5,1])
axes_2[0].set_yticklabels([0,0.5,1])
axes_2[0].yaxis.set_minor_locator(MultipleLocator(0.25))
axes_2[0].yaxis.set_major_locator(MultipleLocator(0.5))
axes_2[0].xaxis.set_major_locator(MultipleLocator(100))
axes_2[0].xaxis.set_minor_locator(MultipleLocator(50))


axes_2[0].set_ylabel(hists_y_label_2,size=MEDIUM_SIZE)
axes_2[2].set_xlabel("$x$ (mm)",size=MEDIUM_SIZE)


for i in range(1,5):
    plt.setp(axes_2[i].get_yticklabels(), visible=False)
    plt.setp(axes_2[i].get_yticklabels(), visible=False)

set_grid(axes_2)
















######################################## Data round


### load data : set up the loading


start_r_2=110
stop_r_2=215

start_2=np.sum(time_sm_2<start_r_2)+900
stop_2=np.sum(time_sm_2<stop_r_2)+900




color=[solid_in_granular_color]*5+[solid_solid_color]+[solid_in_granular_color]*4
labels = [r"$x=5 \pm 0.25$ mm"]+[r"$x={}$".format(i) for i in [int(x_plot[i]) for i in range(1,len(x_plot))]]

for i in range(10):
    axes_3[i].plot(time_sm_2[start_2:stop_2],(eps_xy_sm_2[i][start_2:stop_2])*1e3,color=color[i])
    axes_3[i].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axes_3[i].grid("both")
    y_lim=(np.mean(eps_xy_sm_2[i][start_2:stop_2]))*1e3+np.array([-0.25,0.25])
    axes_3[i].set_ylim(y_lim)
    axes_3[i].yaxis.set_major_locator(MultipleLocator(0.2))
    plt.setp(axes_3[i].get_yticklabels(), visible=False)
    axes_3[i].ticklabel_format(style='plain')
    axes_3[i].annotate(labels[i], xy=(0.01, 0.118), xycoords='axes fraction',
                xytext=(10, 0), textcoords='offset points',
                va="center",size=INSET_SMALL_SIZE)

axes_3[-1].xaxis.set_minor_locator(MultipleLocator(10))
axes_3[-1].xaxis.set_major_locator(MultipleLocator(20))

for i in range(9):
    plt.setp(axes_3[i].get_yticklabels(), visible=False)
    plt.setp(axes_3[i].get_xticklabels(), visible=False)


axes_3[2].text(107.5,0.245,r"$|$",fontsize=10,weight='bold')
axes_3[2].text(104,-0.02,r"200 µm/m",rotation=90,size=INSET_SMALL_SIZE)


#yhere= 1.5e-1
#pos=[(117,yhere),(137,yhere),(157,yhere)]
#tipo=['L','L','L']

#for i in range(len(pos)):
#    axes_3[-1].text(pos[i][0],pos[i][1],tipo[i],size=INSET_SMALL_SIZE)

#tipo=['L','L','SL']
#for i in range(len(pos)):
#    axes_3[-2].text(pos[i][0],pos[i][1],tipo[i],size=INSET_SMALL_SIZE)


# line slipatch
# xs=[0.9725]*2
# ys=[0.48,0.79]
# line2 = mpl.lines.Line2D(xs, ys, lw=1, color='k', alpha=1)
# fig.add_artist(line2)
#
# axes_3[8].text(171,0.8,r"Slipping patch",rotation=-90,size=5)



axes_3[-1].set_xlim([start_r_2,stop_r_2])
axes_3[-1].set_xlabel('temps (s)')
axes_3[4].set_ylabel(r'$\varepsilon_{xy}$',size=MEDIUM_SIZE)

# js=[7,9]
# ti=np.array([104.8,108])
# ys=[
# [-0.07,0.008],
# [0.07,0.14]
# ]
#
# ys=np.array(ys)
# ys=ys+0.01
#
# for i in range(len(js)):
#     axes_3[js[i]].plot(ti,ys[i],c="k",linewidth=1.2)

mean_lc=data_loaded_3["mean_lc"]
manip_num=np.array(data_loaded_3["manip_num"])
LC_there = mean_lc[manip_num==chosen_manip_fig42_sec][0]


axes_3[0].annotate("${}=${:.2f}".format(LC_name_short,LC_there), xy=(0.75, 0.8), xycoords='axes fraction',
                xytext=(10, 0), textcoords='offset points',
                va="center",size=INSET_SMALL_SIZE)



set_grid(axes_3)








##
ell = np.array(ell)

xxs = np.arange(0,15,1)*10

lc_per_ell_round = [np.mean(lc_event[ell == xxs[i]]) if len(lc_event[ell == xxs[i]])!=0 else np.nan for i in range(len(xxs))]

lc_list_round = np.array([-1,-0.5,0,1,2])

#lc_list_round = np.linspace(-1,2,15)



ell_per_lc_round = np.array(
        [np.mean(
            ell[np.logical_and(
                lc_event<lc_list_round[i+1],
                lc_event>lc_list_round[i],
                np.logical_not(from_solid)
            )])/1000
        if len(
            ell[np.logical_and(lc_event<lc_list_round[i+1],lc_event>lc_list_round[i])]
            )!=0
        else np.nan
        for i in range(len(lc_list_round)-1)]
        )

ell_per_lc_round_width = np.array(
        [np.std(
            ell[np.logical_and(
                lc_event<lc_list_round[i+1],
                lc_event>lc_list_round[i],
                np.logical_not(from_solid)
            )])/1000
        if len(
            ell[np.logical_and(lc_event<lc_list_round[i+1],lc_event>lc_list_round[i])]
            )!=0
        else np.nan
        for i in range(len(lc_list_round)-1)]
        )


ell_per_lc_round_solid = np.mean(ell[from_solid])/1000


ell_per_lc_round_width_solid = np.std(ell[from_solid])/1000

lc_solid_mean = -0.9887746801213596# taken from fig 4 np.mean(lc_event[from_solid])
# hard code, ugly




data_fig_4 = np.load(main_folder+"data_for_figures/figure_4.npy",allow_pickle=True).all()

lc_list_nuc                 = data_fig_4["lc_list_nuc"]
ell_per_lc_nuc              = data_fig_4["ell_per_lc_nuc"]
ell_per_lc_nuc_solid        = data_fig_4["ell_per_lc_nuc_solid"]
lc_solid_mean               = data_fig_4["lc_solid_mean"]
ell_per_lc_nuc_width        = data_fig_4["ell_per_lc_nuc_width"]
ell_per_lc_nuc_width_solid  = data_fig_4["ell_per_lc_nuc_width_solid"]










ax_4.errorbar(lc_solid_mean,ell_per_lc_round_solid*1000/30,yerr = ell_per_lc_round_width_solid*1000/30,
                 fmt=" ",capsize=error_bar_width,color="k",
               ecolor=error_bar_color,
               elinewidth=.5,alpha=error_bar_alpha,markeredgewidth=markeredgewidth)

ax_4.scatter((lc_list_round[:-1]+lc_list_round[1:])/2,ell_per_lc_round*1000/30,
        c=main_plot_color,s=scatter_size*2,
        marker="d",zorder=2,
        edgecolors="k",linewidth=0.01)



ax_4.scatter(lc_solid_mean,ell_per_lc_round_solid*1000/30,
        c=solid_solid_color,s=scatter_size*4/3,
        marker="o",zorder=2,
        edgecolors="k",linewidth=0.01)



ax_4.errorbar((lc_list_round[:-1]+lc_list_round[1:])/2,ell_per_lc_round*1000/30,yerr = ell_per_lc_round_width*1000/30,
                 fmt=" ",capsize=error_bar_width,color="k",
               ecolor=error_bar_color,
               elinewidth=.5,alpha=error_bar_alpha,markeredgewidth=markeredgewidth)


"""
ax_4.errorbar(ell_per_lc_nuc*1000/15,
        ell_per_lc_round*1000/30,
        xerr = ell_per_lc_nuc_width*1000/15,
        yerr = ell_per_lc_round_width*1000/30,
        fmt=" ",capsize=error_bar_width,color="k",
        ecolor=error_bar_color,
        elinewidth=.5,alpha=error_bar_alpha,markeredgewidth=markeredgewidth)


ax_4.scatter(ell_per_lc_nuc*1000/15,
        ell_per_lc_round*1000/30,
        c=main_plot_color,s=scatter_size*2,
        marker="d",zorder=2,
        edgecolors="k",linewidth=0.01)


ax_4.errorbar(ell_per_lc_nuc_solid*1000/15,
        ell_per_lc_round_solid*1000/30,
        xerr = ell_per_lc_nuc_width_solid*1000/15,
        yerr = ell_per_lc_round_width_solid*1000/30,
        fmt=" ",capsize=error_bar_width,color="k",
        ecolor=error_bar_color,
        elinewidth=.5,alpha=error_bar_alpha,markeredgewidth=markeredgewidth)


ax_4.scatter(ell_per_lc_nuc_solid*1000/15,
        ell_per_lc_round_solid*1000/30,
        c=solid_solid_color,s=scatter_size*4/3,
        marker="o",zorder=2,
        edgecolors="k",linewidth=0.01)



ax_4.plot([0,4.3],[0,4.3],linestyle = '--',color = "k")


#ax_4.legend()
ax_4.set_xlim(0.8,4.3)
ax_4.set_ylim(0.8,4.3)
#ax_4.set_ylim(23,120)
ax_4.set_xlabel(r"$\frac{\left\langle d_{nuc}\right\rangle}{\ell_{eye}\,/\,2}$",labelpad = 1.5)
ax_4.set_ylabel(r"$\left\langle\ell_{slip}\right\rangle\,/\,\ell_{eye}$")

ax_4.xaxis.set_minor_locator(MultipleLocator(0.5))
ax_4.xaxis.set_major_locator(MultipleLocator(1))
ax_4.yaxis.set_minor_locator(MultipleLocator(0.5))
ax_4.yaxis.set_major_locator(MultipleLocator(1))
#ax_4.yaxis.set_minor_locator(MultipleLocator(25))
#ax_4.yaxis.set_major_locator(MultipleLocator(50))
"""



#ax_4.legend()
ax_4.set_xlim(-1.1,2)
#ax_4.set_ylim(23,120)
ax_4.set_xlabel("${}$".format(LC_name_short))
ax_4.set_ylabel(r"$\left\langle\ell_{slip}\right\rangle\,/\,\ell_{eye}$")

ax_4.xaxis.set_minor_locator(MultipleLocator(0.5))
ax_4.xaxis.set_major_locator(MultipleLocator(1))
ax_4.yaxis.set_minor_locator(MultipleLocator(0.5))
ax_4.yaxis.set_major_locator(MultipleLocator(1))


set_grid(ax_4)





################### Plot adjustment


axes_1[-1].xaxis.labelpad=1
axes_2[2].xaxis.labelpad=1

axes_1[4].yaxis.set_label_coords(-0.05, 0)
axes_2[0].yaxis.set_label_coords(-0.25, 0.5)
ax_4.yaxis.set_label_coords(-0.15, 0.5)




axes_3[-1].xaxis.labelpad=1
#axes_4[2].xaxis.labelpad=1

axes_3[4].yaxis.set_label_coords(-0.05, 0)
#axes_4[0].yaxis.set_label_coords(-0.065*5, 0.5)


#real_tight_layout(fig)
fig.text(0.005,0.98,"a.",size=LETTER_SIZE, weight='bold')
fig.text(.5,0.98,"b.",size=LETTER_SIZE, weight='bold')
fig.text(0.005,0.28,"c.",size=LETTER_SIZE, weight='bold')
fig.text(.72,0.28,"d.",size=LETTER_SIZE, weight='bold')

fig.savefig(main_folder + "Figures_chap_article/figure_5.png",dpi=dpi_global)
fig.savefig(main_folder + "Figures_chap_article/figure_5.pdf")
fig.savefig(main_folder + "Figures_chap_article/figure_5.svg")
plt.close('all')












