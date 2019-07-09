#FEM code for 1D 2nd order steady state equation
#Marcel Frehner, ETH Zurich, 2017
import numpy as np
import matplotlib.pyplot as plt


#GENERAL STUFF

#GEOMETRICAL PARAMETERS
Lx	=	10                                     #length of model [m]
    
#PHYSICAL PARAMETERS
kappa	=	1.0                                 # thermal diffusivity [m2/s]
source	=	1.0                                  # heat source [K/s]
    
#NUMERICAL PARAMETERS
el_tot      =	20                             #elements total
n_tot       = el_tot + 1                    #nodes total
n_per_el	=	2                              #nodes per element
    
#CREATE NUMERICAL GRID
dx      =  Lx/el_tot                     	#distance between two nodes
GCOORD  = np.linspace(0,Lx,n_tot)                 #array of coordinates
	
#LOCAL-TO-GLOBAL MAPPING
EL_N    =   np.array([np.linspace(0,el_tot-1,el_tot,dtype=int),np.linspace(1,el_tot,el_tot,dtype=int)])    #relates local to global node numbers per element
	
#BOUNDARY CONDITIONS
bc_dof  =   np.array([   0 , el_tot], dtype=int)   #dof's to which Dirichlet bc's are assigned
bc_val  =   np.array([   0.0, 0.0   ])     # value for these dof's
    
#INITIALIZATION OF ALL KINDS OF STUFF
KG	=	np.zeros((n_tot,n_tot))   #global stiffness matrix
FG	=	np.zeros((n_tot,1))       #global force vector
   
Kloc    =    (kappa/dx)*np.array([[1, -1],[ -1, 1]]) #local stiffness matrix
Floc    =   source*dx*0.5*np.array([[1],[1]])        #local force vector


for iel in range(0,el_tot):  # ELEMENT LOOP
    n_now   =   EL_N[:,iel]           #which nodes are in the current element?

    #add local matrices to global ones
    #KG[n_now[0]:n_now[1]+1,n_now[0]:n_now[1]+1]     =   KG[n_now[0]:n_now[1]+1,n_now[0]:n_now[1]+1] + Kloc #check np.idx_ 
    #FG[n_now[0]:n_now[1]+1]          =   FG[n_now[0]:n_now[1]+1]+Floc
    KG[np.ix_(n_now,n_now)]     =   KG[np.ix_(n_now,n_now)] + Kloc #check np.idx_ 
    FG[np.ix_(n_now)] = FG[np.ix_(n_now)]+Floc



#APPLY BOUNDARY CONDITIONS


for j,i in enumerate(bc_dof):
    KG[i, : ] = 0.0
    KG[i,i] = 1.0
    FG[i]= bc_val[j]

    

#SOLVER
T	= np.linalg.solve(KG,FG)

    
#ANALYTICAL SOLUTION
x_ana 	=  np.linspace(0,Lx,1000)
T_ana  	=   -1.0/2*source/kappa*x_ana**2 + (1/2*source*Lx/kappa+(bc_val[1]-bc_val[0])/Lx)*x_ana + bc_val[0]

    
#PLOTTING
plt.figure()
plt.plot(x_ana, T_ana, '-', color='red')
plt.plot(GCOORD,T,'o',color='blue')
plt.show()