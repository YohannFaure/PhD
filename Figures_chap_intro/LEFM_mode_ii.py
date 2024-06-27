

main_folder = 'E:/manuscrit_these/'
main_folder = '../'


with open(main_folder + "parameters.py", 'r') as f:
        exec(f.read())




import numpy as np
import scipy.integrate as integrate


def cart2pol(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta


constants={"C_r":1255,"C_s":1345,"C_d":2332,"nu":0.335,"E":5.651e9}




def theta_x(theta,alpha_x):
    """
    Intermediary function
    """
    return(np.arctan(alpha_x*np.tan(theta))%np.pi)

def gamma(theta,v,c):
    """
    Used to compute the fracture energy
    """
    return(np.sqrt(1-(v*np.sin(theta)/c)**2))



def give_me_my_functions(args,PlaneStress=True,constants=constants):
    try :
        C_r=constants["C_r"]
        C_s=constants["C_s"]
        C_d=constants["C_d"]
        nu=constants["nu"]
        E=constants["E"]
    except :
        print("Faulty constants given in full_epsilon, falling back to PMMA")
        C_r=1237
        C_s=1345
        C_d=2332.5
        nu=0.335
        E=5.651e9

    v,y_pos,t_0,Gamma=args
    # compute position
    x=-v*(time-t_0)
    y=np.array([y_pos for _ in x])

    r,theta=cart2pol(x,y)

    # constants
    alpha_s=np.sqrt(1-(v/C_s)**2)
    alpha_d=np.sqrt(1-(v/C_d)**2)
    D=4*alpha_s*alpha_d-(1+alpha_s**2)**2
    #mu=E/(2*(1+nu))
    k_d=1/np.sqrt((1+nu)*alpha_s*v**2/(D*C_s**2))
    if not PlaneStress:
        k_d=k_d/np.sqrt((1+nu)*(1-nu))
    K_s=np.sqrt(Gamma*E)
    K=K_s*k_d

    # functions \Sigma
    def f_yy(theta):
        gamma_s=gamma(theta,v,C_s)
        gamma_d=gamma(theta,v,C_d)
        theta_d=theta_x(theta,alpha_d)
        theta_s=theta_x(theta,alpha_s)
        pref_tot=2*alpha_s*(1+alpha_s**2)/D
        pref_1=1/np.sqrt(gamma_d)
        pref_2=-1/np.sqrt(gamma_s)
        f=pref_tot * ( pref_1 * np.sin(.5*theta_d) + pref_2 * np.sin( .5*theta_s ) )
        return(f)

    def f_xx(theta):
        gamma_s=gamma(theta,v,C_s)
        gamma_d=gamma(theta,v,C_d)
        theta_d= theta_x(theta,alpha_d)
        theta_s=theta_x(theta,alpha_s)
        pref_tot=-2*alpha_s/D
        pref_1=(1+2*alpha_d**2-alpha_s**2)/np.sqrt(gamma_d)
        pref_2=-(1+alpha_s**2)/np.sqrt(gamma_s)
        f=pref_tot * ( pref_1 * np.sin(.5*theta_d) + pref_2 * np.sin( .5*theta_s ) )
        return(f)

    def f_xy(theta):
        gamma_s=gamma(theta,v,C_s)
        gamma_d=gamma(theta,v,C_d)
        theta_d=theta_x(theta,alpha_d)
        theta_s=theta_x(theta,alpha_s)
        pref_tot=1/D
        pref_1=4*alpha_s*alpha_d/np.sqrt(gamma_d)
        pref_2=-((1+alpha_s**2)**2)/np.sqrt(gamma_s)
        f=pref_tot * ( pref_1 * np.cos(.5*theta_d) + pref_2 * np.cos( .5*theta_s ) )
        return(f)

    return(f_xx,f_yy,f_xy)

def full_epsilon_mode_I(time,args,PlaneStress=True,constants=constants):
    """
    This function computes the 2D LEFM solution in function of time and args
        v,y_pos,t_0,Gamma=args
    You can give different constants using the dictionnary constants
        constants={"C_r":1255,"C_s":1345,"C_d":2332,"nu":0.335,"E":5.651e9}
    It returns epsilon_xx, yy and xy
    the different expressions come from Freund, p 163, p 234
    except from Plane Strain K, which comes from brauberg
    """
    try :
        C_r=constants["C_r"]
        C_s=constants["C_s"]
        C_d=constants["C_d"]
        nu=constants["nu"]
        E=constants["E"]
    except :
        print("Faulty constants given in full_epsilon, falling back to PMMA")
        C_r=1237
        C_s=1345
        C_d=2332.5
        nu=0.335
        E=5.651e9

    v,y_pos,t_0,Gamma=args
    # compute position
    x=-v*(time-t_0)
    y=np.array([y_pos for _ in x])

    r,theta=cart2pol(x,y)

    # constants
    alpha_s=np.sqrt(1-(v/C_s)**2)
    alpha_d=np.sqrt(1-(v/C_d)**2)
    D=4*alpha_s*alpha_d-(1+alpha_s**2)**2
    #mu=E/(2*(1+nu))
    k_d=1/np.sqrt((1+nu)*alpha_s*v**2/(D*C_s**2))
    if not PlaneStress:
        k_d=k_d/np.sqrt((1+nu)*(1-nu))
    K_s=np.sqrt(Gamma*E)
    K=K_s*k_d

    # functions \Sigma
    def f_yy(theta):
        gamma_s=gamma(theta,v,C_s)
        gamma_d=gamma(theta,v,C_d)
        theta_d=theta_x(theta,alpha_d)
        theta_s=theta_x(theta,alpha_s)
        pref_tot=1/D
        pref_1=(1+alpha_s**2)*(1+2*alpha_d**2-alpha_s**2)/np.sqrt(gamma_d)
        pref_2=-4*alpha_s*alpha_d/np.sqrt(gamma_s)
        f=pref_tot * ( pref_1 * np.cos(.5*theta_d) + pref_2 * np.cos( .5*theta_s ) )
        return(f)

    def f_xx(theta):
        gamma_s=gamma(theta,v,C_s)
        gamma_d=gamma(theta,v,C_d)
        theta_d= theta_x(theta,alpha_d)
        theta_s=theta_x(theta,alpha_s)
        pref_tot=-1/D
        pref_1=(1+alpha_s**2)**2/np.sqrt(gamma_d)
        pref_2=-4*alpha_s*alpha_d/np.sqrt(gamma_s)
        f=pref_tot * ( pref_1 * np.cos(.5*theta_d) + pref_2 * np.cos( .5*theta_s ) )
        return(f)

    def f_xy(theta):
        gamma_s=gamma(theta,v,C_s)
        gamma_d=gamma(theta,v,C_d)
        theta_d=theta_x(theta,alpha_d)
        theta_s=theta_x(theta,alpha_s)
        pref_tot=2*alpha_d*(1+alpha_s**2)/D
        pref_1=1/np.sqrt(gamma_d)
        pref_2=-1/np.sqrt(gamma_s)
        f=pref_tot * ( pref_1 * np.sin(.5*theta_d) + pref_2 * np.sin( .5*theta_s ) )
        return(f)

    def f_yx(theta):
        return(f_xy(theta))

    def sigma_xx(r,theta,K=K):
        return(K*f_xx(theta)/np.sqrt(2*np.pi*r))

    def sigma_yy(r,theta,K=K):
        return(K*f_yy(theta)/np.sqrt(2*np.pi*r))

    def sigma_xy(r,theta,K=K):
        return(K*f_xy(theta)/np.sqrt(2*np.pi*r))

    def sigma_yx(r,theta,K=K):
        return(sigma_xy(r,theta,K))
    if PlaneStress:
        # Plane stress
        def epsilon_xx(r,theta,K=K):
            return(( sigma_xx(r,theta,K=K) - nu*sigma_yy(r,theta,K=K) )/E)

        def epsilon_yy(r,theta,K=K):
            return(( sigma_yy(r,theta,K=K)-nu*sigma_xx(r,theta,K=K))/E)

        def epsilon_xy(r,theta,K=K):
            return(((1+nu)/E)*sigma_xy(r,theta,K=K))

        def epsilon_yx(r,theta,K=K):
            return(epsilon_xy(r,theta,K=K))
    else:
        # Plane strain
        def epsilon_xx(r,theta,K=K):
            return(( (1-nu**2)*sigma_xx(r,theta,K=K) - nu*(1+nu)*sigma_yy(r,theta,K=K) )/E)

        def epsilon_yy(r,theta,K=K):
            return(( (1-nu**2)*sigma_yy(r,theta,K=K) - nu*(1+nu)*sigma_xx(r,theta,K=K) )/E)

        def epsilon_xy(r,theta,K=K):
            return(((1+nu)/E)*sigma_xy(r,theta,K=K))

        def epsilon_yx(r,theta,K=K):
            return(epsilon_xy(r,theta,K=K))

    # computation
    xx=epsilon_xx(r,theta)
    yy=epsilon_yy(r,theta)
    xy=epsilon_xy(r,theta)
    return(np.array([xx,yy,xy]))


















def full_epsilon(time,args,PlaneStress=True,constants=constants):
    """
    This function computes the 2D LEFM solution in function of time and args
        v,y_pos,t_0,Gamma=args
    You can give different constants using the dictionnary constants
        constants={"C_r":1255,"C_s":1345,"C_d":2332,"nu":0.335,"E":5.651e9}
    It returns epsilon_xx, yy and xy
    the different expressions come from Freund, p 163, p 234
    except from Plane Strain K, which comes from brauberg
    """
    try :
        C_r=constants["C_r"]
        C_s=constants["C_s"]
        C_d=constants["C_d"]
        nu=constants["nu"]
        E=constants["E"]
    except :
        print("Faulty constants given in full_epsilon, falling back to PMMA")
        C_r=1237
        C_s=1345
        C_d=2332.5
        nu=0.335
        E=5.651e9

    v,y_pos,t_0,Gamma=args
    # compute position
    x=-v*(time-t_0)
    y=np.array([y_pos for _ in x])

    r,theta=cart2pol(x,y)

    # constants
    alpha_s=np.sqrt(1-(v/C_s)**2)
    alpha_d=np.sqrt(1-(v/C_d)**2)
    D=4*alpha_s*alpha_d-(1+alpha_s**2)**2
    #mu=E/(2*(1+nu))
    k_d=1/np.sqrt((1+nu)*alpha_s*v**2/(D*C_s**2))
    if not PlaneStress:
        k_d=k_d/np.sqrt((1+nu)*(1-nu))
    K_s=np.sqrt(Gamma*E)
    K=K_s*k_d

    # functions \Sigma
    def f_yy(theta):
        gamma_s=gamma(theta,v,C_s)
        gamma_d=gamma(theta,v,C_d)
        theta_d=theta_x(theta,alpha_d)
        theta_s=theta_x(theta,alpha_s)
        pref_tot=2*alpha_s*(1+alpha_s**2)/D
        pref_1=1/np.sqrt(gamma_d)
        pref_2=-1/np.sqrt(gamma_s)
        f=pref_tot * ( pref_1 * np.sin(.5*theta_d) + pref_2 * np.sin( .5*theta_s ) )
        return(f)

    def f_xx(theta):
        gamma_s=gamma(theta,v,C_s)
        gamma_d=gamma(theta,v,C_d)
        theta_d= theta_x(theta,alpha_d)
        theta_s=theta_x(theta,alpha_s)
        pref_tot=-2*alpha_s/D
        pref_1=(1+2*alpha_d**2-alpha_s**2)/np.sqrt(gamma_d)
        pref_2=-(1+alpha_s**2)/np.sqrt(gamma_s)
        f=pref_tot * ( pref_1 * np.sin(.5*theta_d) + pref_2 * np.sin( .5*theta_s ) )
        return(f)

    def f_xy(theta):
        gamma_s=gamma(theta,v,C_s)
        gamma_d=gamma(theta,v,C_d)
        theta_d=theta_x(theta,alpha_d)
        theta_s=theta_x(theta,alpha_s)
        pref_tot=1/D
        pref_1=4*alpha_s*alpha_d/np.sqrt(gamma_d)
        pref_2=-((1+alpha_s**2)**2)/np.sqrt(gamma_s)
        f=pref_tot * ( pref_1 * np.cos(.5*theta_d) + pref_2 * np.cos( .5*theta_s ) )
        return(f)

    def f_yx(theta):
        return(f_xy(theta))

    def sigma_xx(r,theta,K=K):
        return(K*f_xx(theta)/np.sqrt(2*np.pi*r))

    def sigma_yy(r,theta,K=K):
        return(K*f_yy(theta)/np.sqrt(2*np.pi*r))

    def sigma_xy(r,theta,K=K):
        return(K*f_xy(theta)/np.sqrt(2*np.pi*r))

    def sigma_yx(r,theta,K=K):
        return(sigma_xy(r,theta,K))
    if PlaneStress:
        # Plane stress
        def epsilon_xx(r,theta,K=K):
            return(( sigma_xx(r,theta,K=K) - nu*sigma_yy(r,theta,K=K) )/E)

        def epsilon_yy(r,theta,K=K):
            return(( sigma_yy(r,theta,K=K)-nu*sigma_xx(r,theta,K=K))/E)

        def epsilon_xy(r,theta,K=K):
            return(((1+nu)/E)*sigma_xy(r,theta,K=K))

        def epsilon_yx(r,theta,K=K):
            return(epsilon_xy(r,theta,K=K))
    else:
        # Plane strain
        def epsilon_xx(r,theta,K=K):
            return(( (1-nu**2)*sigma_xx(r,theta,K=K) - nu*(1+nu)*sigma_yy(r,theta,K=K) )/E)

        def epsilon_yy(r,theta,K=K):
            return(( (1-nu**2)*sigma_yy(r,theta,K=K) - nu*(1+nu)*sigma_xx(r,theta,K=K) )/E)

        def epsilon_xy(r,theta,K=K):
            return(((1+nu)/E)*sigma_xy(r,theta,K=K))

        def epsilon_yx(r,theta,K=K):
            return(epsilon_xy(r,theta,K=K))

    # computation
    xx=epsilon_xx(r,theta)
    yy=epsilon_yy(r,theta)
    xy=epsilon_xy(r,theta)
    return(np.array([xx,yy,xy]))



























def full_epsilon_alpha_bis(time,args,PlaneStress=False):
    """
    This is shit
    """
    v,y_pos,t_0,Gamma,angle,e=args
    distance=v*time
    dd=np.mean(np.diff(distance))
    d_max=(e/np.sin(angle))
    n_inc=int(d_max/dd)
    argsbis=[v,y_pos,t_0,Gamma]
    result=full_epsilon(time,argsbis,PlaneStress=PlaneStress)

    def rolling_sum(a, n=4) :
        ret = np.cumsum(a, axis=1, dtype=float)
        ret[:, n:] = ret[:, n:] - ret[:, :-n]
        return(ret/n)

    result = rolling_sum(result, n_inc)
    return(result)


def full_epsilon_alpha(time,args,PlaneStress=False):
    """
    Computes epsilon with LEFM solutions but considers the thickness of the bloc
    and an angle between the ruture front and the propagation axis
    """
    v,y_pos,t_0,Gamma,angle,e=args
    angle=np.pi*angle/2
    # number of points to compute the integral over thickness
    # this number has to go up with alpha, because alpha controls the delay
    # between first and last arrived fonts.
    n_thickness=int(10+100*np.abs(1-2*angle/np.pi)**2)
    thickness_accountant=np.arange(0,1,1/n_thickness)

    # take into account delay between the front and back of the rupture
    # this delay translates into a change of alpha in the lefm solution
    x_max=(e/np.tan(angle))
    x_decal_arr=x_max*thickness_accountant
    t_0_arr=x_decal_arr/v+t_0

    # take into account thickness e
    z=e*thickness_accountant
    y_pos_arr=np.sqrt(y_pos**2+z**2+x_decal_arr**2)
    # Not sure if I have to take x_decal_arr**2...


    # Computing the discrete integral
    results=full_epsilon(time,[v,y_pos_arr[0],t_0_arr[0],Gamma],PlaneStress)
    for i in range(1,n_thickness):
        results+=full_epsilon(time,[v,y_pos_arr[i],t_0_arr[i],Gamma],PlaneStress)
    results=results/n_thickness
    return(results)
































### Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from matplotlib import rc
from scipy.optimize import curve_fit
from matplotlib.widgets import Slider, Button
import os




### Constants
C_r=1255
C_s=1345
C_d=2332
nu=0.335
E=4e9
constants={"C_r":C_r,"C_s":C_s,"C_d":C_d,"nu":nu,"E":E}

time = np.arange(-0.005,0.005,2e-8)

### Fitting function


#set bounds
bounds=[
[ 10     ,C_r  ], #v
[ 1e-5 ,0.02 ], #y_pos
[min(time),max(time) ], #t_0
[ 1e-5    ,10     ]] #Gamma
bounds=np.array(bounds)
bounds=bounds.transpose()

# set proportion of the data to take into account

names=['v     ','y_pos ','t_0   ','Gamma ']



### Optimization computation

args=np.array([500,0.003,0,1])

### Compute results

xx,yy,xy = full_epsilon(time,args,PlaneStress=False,constants=constants)
xxp,yyp,xyp = full_epsilon(time,args,PlaneStress=True,constants=constants)
v,y_pos,t_0,Gamma=args





###


args[-1]=2000
args[1]=0.001

xx,yy,xy = full_epsilon(time,args,PlaneStress=False,constants=constants)
xxp,yyp,xyp = full_epsilon(time,args,PlaneStress=True,constants=constants)
v,y_pos,t_0,Gamma=args

fig, axes = plt.subplots(1, 3,sharex=True,sharey=False)
fig.set_size_inches(size_fig_LEFM)


line4,=axes[0].plot(1000000 * time,1000 * xxp,color=main_plot_color)
line5,=axes[1].plot(1000000 * time,1000 * yyp,color=main_plot_color)
line6,=axes[2].plot(1000000 * time,1000 * xyp,color=main_plot_color)

axes[1].set_xlabel("temps (µs)")
axes[0].set_ylabel(r'$\varepsilon_{xx}$ (mm/m)')
axes[1].set_ylabel(r'$\varepsilon_{yy}$ (mm/m)')
axes[2].set_ylabel(r'$\varepsilon_{xy}$ (mm/m)')



for i in range(3):
    axes[i].xaxis.set_minor_locator(MultipleLocator(25))
    axes[i].xaxis.set_major_locator(MultipleLocator(50))
    axes[i].yaxis.set_minor_locator(MultipleLocator(1))
    axes[i].yaxis.set_major_locator(MultipleLocator(2))




set_grid(axes)

axes[0].set_xlim((-100,100))

plt.tight_layout()

fig.savefig(main_folder + "Figures_chap_intro/LEFM_plot_ii.png",dpi=dpi_global)
fig.savefig(main_folder + "Figures_chap_intro/LEFM_plot_ii.pdf")
fig.savefig(main_folder + "Figures_chap_intro/LEFM_plot_ii.svg")
fig.show()
plt.close('all')










