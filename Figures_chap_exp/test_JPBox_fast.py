## Imports
# Science
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from scipy import interpolate
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


main_folder = 'E:/manuscrit_these/'
main_folder = '../'

with open(main_folder + "parameters.py", 'r') as f:
        exec(f.read())

# # custom file
# try :
#     from Python_DAQ import *
# except :
#     from DAQ_Python.Python_DAQ import *
#
# with open(main_folder + "parameters.py", 'r') as f:
#         exec(f.read())
#


### load data : set up the loading
# main folder
#loc_folder_1="E:/2023-2024/2023-12-18-JPBox/"
#loc_folder_2="E:/2023-2024/2023-07-11-manips-10-voies/manip_14/"

loc_folder_1=main_folder + "data_for_figures/noise_gages/jpbox/"
loc_folder_2=main_folder + "data_for_figures/noise_gages/jerubox/"

loc_file="event-001.npy"
roll_smooth=1
start=0


##

fast1=1000*np.load(loc_folder_1+loc_file)


fast_time1=np.arange(0,len(fast1))
fast1=smooth(fast1, n=roll_smooth)
fast_time1=smooth(fast_time1, n=roll_smooth)

fast2=1000*np.load(loc_folder_2+loc_file)
fast_time2=np.arange(0,len(fast2))

fast2=smooth(fast2, n=roll_smooth)
fast_time2=smooth(fast_time2, n=roll_smooth)

test1=(fast1.T-np.mean(fast1,axis=-1)).T
test2=(fast2.T-np.mean(fast2,axis=-1)).T

fast1=test1
fast2=test2


##

std2_1=np.std(fast2[0:15],axis=-1)
std2_2=np.std(fast2[16:31],axis=-1)
std1_1=np.std(fast1[0:12],axis=-1)
std1_2=np.std(fast1[[12,13,14,16,17,18,19,20,21,22,23,24]],axis=-1)
std1_3=np.std(fast1[[25,26,27,28,29,30,32,33,34,35,36,37]],axis=-1)
std1_4=np.std(fast1[[38,39,40,41,42,43,44,45,46,48,49,50]],axis=-1)



##

fig, axes=plt.subplots(1,2,sharex=True)
fig.set_size_inches(size_test_JPBox)

x1=np.arange(1,13,1)
x2=np.arange(1,16,1)

axes[0].plot(x2,std2_2,c=(0,150/255,200/255),label="carte DIP (originale)",marker="+")
axes[0].plot(x2,std2_1,c=(0,50/255,250/255),label="carte DIP (copie)",marker="+")
axes[0].plot(x1,std1_1[::-1],c=(230/255,0,35/255),label='cartes SOIC',marker="+")
axes[0].plot(x1,std1_2[::-1],c=(230/255,0,35/255),marker="+")
axes[0].plot(x1,std1_3[::-1],c=(230/255,0,35/255),marker="+")
axes[0].plot(x1,std1_4[::-1],c=(230/255,0,35/255),marker="+")




axes[0].set_xlabel("position dans la boucle")

axes[0].set_ylabel("bruit après amplification (mV)")

axes[0].legend(facecolor = (1,1,1,1),framealpha=1)


axes[1].plot(x2,1000*std2_2/500,c=(0,150/255,200/255),label="Jeru",marker="+")
axes[1].plot(x2,1000*std2_1/500,c=(0,50/255,250/255),label="Copie Jeru",marker="+")
axes[1].plot(x1,1000*std1_1[::-1]/2000,c=(230/255,0,35/255),label='JPBox',marker="+")
axes[1].plot(x1,1000*std1_2[::-1]/2000,c=(230/255,0,35/255),marker="+")
axes[1].plot(x1,1000*std1_3[::-1]/2000,c=(230/255,0,35/255),marker="+")
axes[1].plot(x1,1000*std1_4[::-1]/2000,c=(230/255,0,35/255),marker="+")

axes[1].set_ylabel("Bruit normalisé par le gain (µV)")




axes[0].set_ylim([0,35])
axes[0].set_xlim([0,16])
axes[1].set_ylim([0,1000*35/500])


axes[0].yaxis.set_major_locator(MultipleLocator(10))
axes[0].yaxis.set_minor_locator(MultipleLocator(5))
axes[1].yaxis.set_major_locator(MultipleLocator(20))
axes[1].yaxis.set_minor_locator(MultipleLocator(10))

axes[0].xaxis.set_major_locator(MultipleLocator(5))
axes[0].xaxis.set_minor_locator(MultipleLocator(2.5))

set_grid(axes)
fig.tight_layout()





###

plt.savefig(main_folder + "Figures_chap_exp/test_JPBox_fast.png",dpi=1200)
plt.savefig(main_folder + "Figures_chap_exp/test_JPBox_fast.pdf")
plt.savefig(main_folder + "Figures_chap_exp/test_JPBox_fast.svg")
plt.close()













