main_folder = '../'

with open(main_folder + "parameters.py", 'r') as f:
        exec(f.read())




E=3e9
sigma_y=1.2e7

eps=np.linspace(0,1e-2,10000)

def f_1(eps,E=E,sigma_y = sigma_y):
    line = eps * E
    line[line>sigma_y]=sigma_y
    return(line)



def f_2(x,E=E,sigma_y = sigma_y):
    sigma_f = 1.165 * sigma_y

    text = sigma_f * np.tanh(E*x/sigma_f )
    return(text)





fig,ax=plt.subplots(1)
fig.set_size_inches(size_fig_yield)

ax.plot(1000 * eps,1e-6*f_2(eps),label="réponse typique")
ax.plot(1000 * eps,1e-6*f_1(eps),label = "modèle")
ax.axhline(1e-6*sigma_y,linestyle = "--",color="k")
ax.plot(1000 * eps[E*eps<sigma_y*1.5]+0.002*1000,1e-6*E*eps[E*eps<sigma_y*1.5],linestyle = ":",color="k")#,label="seuil  $\Delta\ell = 0.2\%$")


ax.xaxis.set_minor_locator(MultipleLocator(2.5))
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(2.5))
ax.yaxis.set_major_locator(MultipleLocator(5))


ax.set_xlim([0,10])
ax.set_ylim([0,14])
ax.set_xlabel(r"$\varepsilon$ (mm/m)")
ax.set_ylabel(r"$\sigma$ (MPa)")
ax.legend(facecolor = (1,1,1,1),framealpha=1)

set_grid(ax)



ax.annotate("$\sigma_{y}$",(-1,sigma_y*1e-6-0.2),size = SMALL_SIZE,annotation_clip=False)
ax.annotate(r"$\Delta\varepsilon=0.2\%$",(1.1,-1.5),size = SMALL_SIZE,annotation_clip=False)


xs = [0.7,1.7,1.7,0.7]
ys = [E*0.5e-9,E*0.5e-9,E*1.5e-9,E*0.5e-9]
ax.plot(xs,ys,c="r")
ax.annotate("1",(1.1,E*0.2e-9),size = SMALL_SIZE)
ax.annotate("E",(1.8,E*0.9e-9),size = SMALL_SIZE)
fig.subplots_adjust(bottom=0.16, top=0.99,right=0.98,left=0.13)



plt.savefig(main_folder + "Figures_chap_intro/yield.png",dpi=600)
plt.savefig(main_folder + "Figures_chap_intro/yield.pdf",dpi=600)
plt.savefig(main_folder + "Figures_chap_intro/yield.svg",dpi=600)
plt.close()


