###
main_folder = 'E:/manuscrit_these/'
main_folder = '../'

with open(main_folder + "parameters.py", 'r') as f:
        exec(f.read())




loaded_data = np.load(main_folder + "data_for_figures/figure_S2.npy",allow_pickle= True).all()
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


def creep_plot(x,creep,ax,labels=True,suptitle=False,color="blue",alpha=1,marker=scatter_marker,s=scatter_size,vlines=False ):
    if vlines:
        for i in range(len(x)):
            ax.plot([x[i],x[i]],[creep[i],100],
                    color=dashed_color,linestyle=dash_line_style,
                    linewidth=0.1,alpha=0.2)

    ax.plot(x,creep,color=color,alpha=alpha)
    ax.scatter(x,creep,marker=marker ,s=s ,c=color,alpha = 1 ,zorder=10,edgecolors="k",linewidth=0.01)

    ax.grid(True,which="both")

    if labels:
        ax.set_ylabel(sliding_perc_name_short )

    if suptitle:
        ax.set_title(suptitle)




def mu_plot(t,mu,ax,timings=None,linecolor=main_plot_color):
    a=next((i for i, b in enumerate(mu[7000:]>0.05) if b), None)+7000
    start = max(a-5200,0)
    ax.plot(t[start:]-t[start],mu[start:],color=linecolor )


def ramp(l,v,f=100):
    return(v*np.arange(0,l,1)/f)



##
def findevents(signal, sensit):
    typical_wait_time = 60
    # signal = np.convolve(signal, np.ones(5)/5, mode='valid')  # Smoothing signal, if needed
    diff = np.abs(signal[2:] - signal[:-2])
    m = np.mean(diff)
    diff -= m
    index = np.where(diff > sensit)[0]
    index = np.append(index, index[-1] + 2 * typical_wait_time)
    diffindex = np.abs(-index[1:] + index[:-1])
    found_events = [index[0]]
    found_events.extend(index[np.where(diffindex > typical_wait_time)[0] + 1])
    found_events.pop()  # Remove the last element
    return np.array(found_events)

# Assuming the variables such as avgtop, avgbot, xyresolution, tInit, tEnd, detection_grain,
# listAllCircles, sensit, space_after_event_temp, space_before_event, removelast,
# avgtopleft, avgbotleft, avgtopright, avgbotright, avgtopcenter, avgbotcenter, realtime
# are already defined and initialized in your Python environment



totslip = -data_cam_cut
realtime= np.arange(0,len(data_cam_cut))/100-0.21-5

found_events = findevents(data_cam_cut, 0.01)


if len(found_events) > 1:
    space_after_event_temp = min(np.fix(np.diff(found_events) / 10))
else:
    space_after_event_temp = 50

space_after_event = max(min(50, space_after_event_temp), 15)
space_before_event = 5


##

delta = np.sum(totslip[found_events + space_after_event] - totslip[found_events - space_before_event])

delta_tot = totslip[-1] - totslip[0]

creeped_distance = delta_tot - delta


###
# Compute creep proportion to seek convergence
creep = totslip

delta = []
delta_tot = []
totslip_partial = []

for i in range(len(found_events)):
    delta.append(np.sum(creep[found_events[0:i+1] + space_after_event] - creep[found_events[0:i+1] - space_before_event]))
    delta_tot.append(creep[found_events[i] + space_after_event] - creep[0])

    totslip_partial.append(abs(np.mean(creep[found_events[i] + int(space_after_event/2):found_events[i] + space_after_event]) ) )

creeped_distance_frac = (np.array(delta_tot) - np.array(delta)) / np.array(totslip_partial)
creeped_distance = np.array(delta_tot) - np.array(delta)





## FIGURE

time_lim_1=[-1,51]
time_lim_2=[-0.5,55]



fig = plt.figure(layout=None)
fig.set_size_inches(size_fig_S2_res_only)

nrows=12


gs = fig.add_gridspec(nrows=1, ncols=2,left=0.08)

axes=np.array([None,None])

axes[0]=fig.add_subplot(gs[0])
axes[1]=fig.add_subplot(gs[1], sharex = axes[0])



# TOP LEFT

axes[0].plot(time_simul_ramp,-data_simul_ramp,label="instructions",zorder=3)
axes[0].plot(time_prof_10,-data_prof_10,label="profilomètre")
axes[0].plot(time_cam_ramp,-data_cam_ramp_pix,label="suivi pixel",linestyle="solid",alpha=.5,c=u'#d62728')
axes[0].plot(time_cam_ramp,-data_cam_ramp,label="suivi sous-pixel",c=u'#2ca02c')


axes[0].plot(time_simul_ramp,-data_simul_ramp+0.005,linewidth=.5,c="k",linestyle=':')
axes[0].plot(time_simul_ramp,-data_simul_ramp-0.005,linewidth=.5,c="k",linestyle=':')

rect = mpl.patches.Rectangle((20,0.200),5,0.050,facecolor='None',edgecolor="r",linewidth=2,zorder=4)
axes[0].add_artist(rect)
axes[0].xaxis.set_minor_locator(MultipleLocator(10))
axes[0].xaxis.set_major_locator(MultipleLocator(20))
axes[0].yaxis.set_minor_locator(MultipleLocator(0.1))
axes[0].yaxis.set_major_locator(MultipleLocator(0.2))

axes[0].set_xlim([0,58])
axes[0].set_ylim([0,0.580])
axes[0].legend(facecolor='white', framealpha=1)
set_grid(axes[0])
axes[0].set_ylabel('x (mm)')

coords = [0.33,   0.2,    0.14,  0.3] #left, bottom, width, height
axin00=fig.add_axes(coords)



axin00.plot(time_simul_ramp,-data_simul_ramp,label="simul",zorder=4,linewidth =1)
axin00.plot(time_prof_10,-data_prof_10,label="profilo")
axin00.plot(time_cam_ramp,-data_cam_ramp,label="camera")

axin00.plot(time_simul_ramp, -data_simul_ramp+0.005, linewidth=.5, c="k",linestyle=':')
axin00.plot(time_simul_ramp, -data_simul_ramp-0.005, linewidth=.5, c="k",linestyle=':')

axin00.set_xlim([20,25])
axin00.set_ylim([0.200,0.250])
axin00.xaxis.set_minor_locator(MultipleLocator(1))
axin00.xaxis.set_major_locator(MultipleLocator(2))
axin00.yaxis.set_minor_locator(MultipleLocator(0.010))
axin00.yaxis.set_major_locator(MultipleLocator(0.020))


axin00.grid(False)
axin00.set_facecolor("w")
set_up_inset(axin00,linewidth = None)


# bottom left




# for ramp
min_of_max = min([max(time_simul_ramp),max(time_prof_10),max(time_cam_ramp)])
max_of_min = max([min(time_simul_ramp),min(time_prof_10),min(time_cam_ramp)])

mask_simul = np.logical_and(time_simul_ramp>max_of_min,time_simul_ramp<min_of_max)
mask_prof = np.logical_and(time_prof_10>max_of_min,time_prof_10<min_of_max)
mask_cam = np.logical_and(time_cam_ramp>max_of_min,time_cam_ramp<min_of_max)


time_shared= time_simul_ramp[mask_simul]
compar_simul = -data_simul_ramp[mask_simul]+data_cam_ramp[mask_cam]
compar_simul -= np.mean(compar_simul)
compar_prof = -data_prof_10[mask_prof]+data_cam_ramp[mask_cam]
compar_prof -= np.mean(compar_prof)

axes[1].plot(time_shared,1000*compar_simul,label="$x_{ref}$ from instructions",zorder=3)
axes[1].plot(time_shared,1000*compar_prof,label="$x_{ref}$ from profilometer")



axes[1].yaxis.set_minor_locator(MultipleLocator(5))
axes[1].yaxis.set_major_locator(MultipleLocator(10))

axes[1].set_ylim([-17,17])
axes[1].axhline(5,linestyle = ":", c="k")
axes[1].axhline(-5,linestyle = ":", c="k")

#axes[1].legend(facecolor='white', framealpha=1)
set_grid(axes[1])

axes[0].set_xlabel("temps (s)")
axes[1].set_ylabel("$x-x_{ref}$ (µm)")



# adjust spacing
plt.subplots_adjust(wspace=0.22,hspace=0)


axes[0].yaxis.set_label_coords(-0.11, 0.5)
axes[1].yaxis.set_label_coords(-0.09, 0.5)











fig.subplots_adjust(left=0.11, right=0.99, top=0.95, bottom=0.14)


y=0.95
fig.text(0.01,y,"a",size=LETTER_SIZE, weight="bold")
fig.text(.505,y,"b",size=LETTER_SIZE, weight="bold")


plt.savefig(main_folder + "Figures_chap_exp/figure_S2_res_only.png",dpi=1200)
plt.savefig(main_folder + "Figures_chap_exp/figure_S2_res_only.pdf")
plt.savefig(main_folder + "Figures_chap_exp/figure_S2_res_only.svg")

#plt.show()
plt.close('all')










