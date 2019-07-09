#FEM code for 1D 2nd order steady state equation
#Marcel Frehner, ETH Zurich, 2017
import numpy as np
import matplotlib.pyplot as plt



def kappa_loc(el,el_tot, kappa_0):
    x = el/el_tot

    kappa = kappa_0 * (x-0.5)**2.0+0.5**2.0 

    return kappa_0

def source_loc(el,el_tot, source_0):

    return source_0


def T_analytical(t, Tmax, sigma, kappa,x):

    T = ((Tmax)/(np.sqrt(1+4*t*kappa/(sigma**2.0))))*np.exp((-x**2.0)/(sigma**2.0+4*t*kappa))


    return T

#GENERAL STUFF

#GEOMETRICAL PARAMETERS
xmin = -20
Lx	=	40                                   #length of model [m]
    
#PHYSICAL PARAMETERS
#kappa	=	1.0                                 # thermal diffusivity [m2/s]
#source	=	1.0                                  # heat source [K/s]
    
#NUMERICAL PARAMETERS
el_tot      =	200                     #elements total
n_tot       = el_tot + 1                    #nodes total
n_per_el	=	2                              #nodes per element
    
#CREATE NUMERICAL GRID

dx = np.empty((el_tot))
dx[:]=np.nan
dx_0 = Lx/el_tot
i = 0 
while i < el_tot/4:
    dx[i]= dx_0/1
    i = i + 1

dx_new = (Lx - i*dx_0/1)/(el_tot - i)

while i < el_tot:
    dx[i] = dx_new
    i = i + 1
##############

GCOORD = np.empty((n_tot))
GCOORD[:] = np.nan
for i in range(0,len(GCOORD)):
    if i == 0:
        GCOORD[i]=xmin
    else:
        GCOORD[i]=GCOORD[i-1]+dx[i-1]


dt = 0.5
t_final = 10.0
	
#LOCAL-TO-GLOBAL MAPPING
EL_N    =   np.array([np.linspace(0,el_tot-1,el_tot,dtype=int),np.linspace(1,el_tot,el_tot,dtype=int)])    #relates local to global node numbers per element
	
#BOUNDARY CONDITIONS
bc_dof  =   np.array([   0 , el_tot], dtype=int)   #dof's to which Dirichlet bc's are assigned
bc_val  =   np.array([   0.0, 0.0   ])     # value for these dof's
    
#INITIALIZATION OF ALL KINDS OF STUFF
LG	=	np.zeros((n_tot,n_tot))   #global stiffness matrix
RG  =   np.zeros((n_tot,n_tot))  #global right hand side matrix
FG	=	np.zeros((n_tot,1))       #global force vector
T  =    np.reshape(T_analytical(t=0, Tmax=100, sigma=1.0, kappa=1.0,x=GCOORD),np.shape(FG))     #temperature vector


#Integration definition of variables
n_int = 2 #Number of integration points
xsi = np.array([np.sqrt(1.0/3), np.sqrt(1.0/3)])
weight = np.array([1.0, 1.0])





for iel in range(0,el_tot):  # ELEMENT LOOP

    ############INTEGRATION##################
    N1 = np.array([1.0, 0 ])  #shape function N1 evaluated at -1 a nd 1
    N2 = np.array([0, -1.0 ])  
    dN1 = np.array([-1.0/2, -1.0/2 ]) #derivative of shape function evaluated at -1 and 1 
    dN2 = np.array([1.0/2, 1.0/2 ])

    jac = Lx/2.0  #Jacobian - here only a number






    #########################################

    kappa = kappa_loc(iel,el_tot,1.0)
    source = source_loc(iel,el_tot,0.0)

    dx_loc = dx[iel]

    Kloc =  (kappa/dx_loc)*np.array([[1, -1],[ -1, 1]]) #local stiffness matrix
    Floc =  source*dx_loc*0.5*np.array([[1],[1]])        #local force vector
    Mloc = dx_loc * np.array([[1.0/3, 1.0/6],[ 1.0/6, 1.0/3]])

    Lloc = Mloc/dt + Kloc
    Rloc = Mloc/dt 

    n_now   =   EL_N[:,iel]           #which nodes are in the current element?

    LG[np.ix_(n_now,n_now)]     =   LG[np.ix_(n_now,n_now)] + Lloc 
    RG[np.ix_(n_now,n_now)]     =   RG[np.ix_(n_now,n_now)] + Rloc 
    FG[np.ix_(n_now)] = FG[np.ix_(n_now)]+Floc


time = 0.0 

while time < t_final:  #main time step loop starts

#APPLY BOUNDARY CONDITIONS


    b = np.matmul(RG,T)+FG  #forming the right hand side vector b

    
    for j,i in enumerate(bc_dof):   #boundary conditions for fixed temperature, for Neumann: comment this block as it is the default bc-condition by construction 
        LG[i, : ] = 0.0
        LG[i,i] = 1.0
        b[i]= bc_val[j]
    
    #SOLVER
    T   = np.linalg.solve(LG,b)   #new temperature

    time = time + dt

    #PLOTTING

    #plt.plot(x_ana, T_ana, '-', color='red')


x_ana   =  np.linspace(xmin,xmin+Lx,100)
T_ana = T_analytical(time, Tmax=100, sigma=1.0, kappa=1.0,x = x_ana)
    
plt.figure()
plt.plot(GCOORD,T,'o',color='blue')
plt.plot(x_ana, T_ana, '-', color='red')
plt.show()


    
