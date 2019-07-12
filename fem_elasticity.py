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

#Shape functions and their derivatives
def N(xsi,eta,i):
    if i == 1:
        return 0.25*(1-xsi)*(1-eta)
    elif i == 2:
        return 0.25*(1-xsi)*(1+eta)
    elif i == 3:
        return 0.25*(1+xsi)*(1+eta)
    elif i == 4:
        return 0.25*(1+xsi)*(1-eta)
    else:
        raise ValueError('Variable i must be 1 or 2')

def dNdxsi(xsi,eta,i):
    if i == 1:
        return -0.25*(1-eta)
    elif i == 2:
        return -0.25*(1+eta)
    elif i == 3:
        return 0.25*(1+eta)
    elif i == 4:
        return 0.25*(1-eta)
    else:
        raise ValueError('Variable i must be 1 or 2')

def dNdeta(xsi,eta,i):
    if i == 1:
        return -0.25*(1-xsi)
    elif i == 2:
        return 0.25*(1-xsi)
    elif i == 3:
        return 0.25*(1+xsi)
    elif i == 4:
        return -0.25*(1+xsi)
    else:
        raise ValueError('Variable i must be 1 or 2')

def create_gnum(n_el_x, n_el_y):
    n_el = n_el_x*n_el_y    
    gnum = np.empty((4,0),dtype=int)

    num = 1
    for i in range(0,n_el):
        if num % (n_el_x+1) == 0:  
            num = num + 1
        col = np.array([num, num+n_el_x+1, num+n_el_x+2, num+1]).reshape(4,1)
        gnum = np.append(gnum, col, axis=1)
        num = num + 1
        
    return gnum-1

def create_nf(n_nodes,dof):
#Relationship between nodes and equation numbers (dof is degree of freedom at each node)
    nf = np.empty((dof,0))
    num = 1
    for i in range(0,n_nodes):
        col = np.array([[num,num+1]]).reshape(2,1)
        nf = np.append(nf,col,axis=1)
        num = num+2

    return nf-1

def create_gg   (gnum, nf):
    n_lines = len(gnum[:,0])*len(nf[:,0])
    gg = np.empty((n_lines,0))

    for i in range(0,len(gnum[0,:])):
        col = np.empty((0,1))
        idxs = gnum[:,i] 
        for idxs_i in idxs:
            col = np.append(col,nf[:,idxs_i].reshape((dof,1)),axis=0)

        gg = np.append(gg,col,axis=1)

    return gg





#GENERAL STUFF

#GEOMETRICAL PARAMETERS
xmin = 0
Lx = 20  #length of model in direction x [m]
Ly = 20 #length of model in direction y
    
#PHYSICAL PARAMETERS
#kappa	=	1.0                                 # thermal diffusivity [m2/s]
#source	=	1.0                                  # heat source [K/s]
    
#NUMERICAL PARAMETERS

dim = 2 #dimension
dof = 2 #degree of freedom at nodes
n_el_x = 3#number of elements in a row
n_el_y = 3 #number of elements in a column

el_tot = n_el_x * n_el_y #total number of elements              #elements total
n_nodes_x = n_el_x + 1 #number of nodes in a row
n_nodes_y = n_el_y + 1 #number of nodes in a column
n_per_el	=	4 #nodes per element

n_nodes_tot = n_nodes_x*n_nodes_y




'''
    
#CREATE NUMERICAL GRID
GCOORD = np.empty((0,n_nodes_tot))

xv,yv = np.meshgrid(np.linspace(0,Lx,n_nodes_x),np.linspace(0,Ly,n_nodes_y))
xv = np.ndarray.flatten(xv).reshape((1,n_nodes_tot))
yv = np.ndarray.flatten(yv).reshape((1,n_nodes_tot))

GCOORD = np.append(GCOORD,xv,axis=0)
GCOORD = np.append(GCOORD,yv,axis=0)

#time paramaters
dt = 0.5
t_final = 50
	
#LOCAL-TO-GLOBAL MAPPING
g_num  =   create_gnum(n_el_x=n_el_x,n_el_y=n_el_y)    #relates local to global node numbers per element

	
#BOUNDARY CONDITIONS
bc_dof  =   np.ndarray.flatten(np.array([np.linspace(0,n_el_x,n_nodes_x),np.linspace(n_nodes_tot-1-n_el_x,n_nodes_tot-1,n_nodes_x)], dtype=int))   #dof's to which Dirichlet bc's are assigned
bc_val  =   100.0    # value for these dof's
    
#INITIALIZATION OF ALL KINDS OF STUFF
LG	=	np.zeros((n_nodes_tot,n_nodes_tot))   #global stiffness matrix
RG  =   np.zeros((n_nodes_tot,n_nodes_tot))  #global right hand side matrix
FG	=	np.zeros((n_nodes_tot,1))       #global force vector
T  =    np.zeros((n_nodes_tot,1))   #temperature vector


#Integration definition of variables
n_ip = 4 #Number of integration points
xsi = np.array([-np.sqrt(1.0/3), np.sqrt(1.0/3)])
eta = np.array([-np.sqrt(1.0/3),np.sqrt(1.0/3)])

weight = np.array([1.0, 1.0])



############INTEGRATION##################
for iel in range(0,el_tot):  # ELEMENT LOOP

    kappa = kappa_loc(iel,el_tot,1.0)
    source = source_loc(iel,el_tot,0.0)

    D = np.array([[kappa,0],[0,kappa]])

    
    
    #Initialize the M,K,F matrices
    Mloc = np.zeros((n_per_el,n_per_el))
    Kloc = np.zeros((n_per_el,n_per_el))
    Floc = np.zeros((n_per_el,1))

    
    #Start a loop over all integration points
    for i, xsi_i in enumerate(xsi):
        for j, eta_i in enumerate(eta):
            N_l = np.array([N(xsi_i,eta_i,1),N(xsi_i,eta_i,2),N(xsi_i,eta_i,3),N(xsi_i,eta_i,4)]).reshape(1,4)

            dN_l = np.empty((0,4))

            dNdxsi_l = np.array([dNdxsi(xsi_i,eta_i,1),dNdxsi(xsi_i,eta_i,2),dNdxsi(xsi_i,eta_i,3),dNdxsi(xsi_i,eta_i,4)]).reshape(1,4)# derivative of local shape function at this integration point
            dNdeta_l = np.array([dNdeta(xsi_i,eta_i,1),dNdeta(xsi_i,eta_i,2),dNdeta(xsi_i,eta_i,3),dNdeta(xsi_i,eta_i,4)]).reshape(1,4)

            dN_l = np.append(dN_l,dNdxsi_l,axis=0)
            dN_l = np.append(dN_l,dNdeta_l,axis=0)

            #calculate the Jacobian with equation 5.23 in the script
            n_now = g_num[:,iel] #which nodes are in the current element?
            gcoord_nodes = GCOORD[:,n_now].transpose() #what are the corresponding x and y coordinates of the nodes? transpose to have same form as in script (4x2)

            jac = np.matmul(dN_l,gcoord_nodes) #Calculate the Jacobian matrix

            dN_g = np.matmul(np.linalg.inv(jac),dN_l) #Convert the derivatives from the local coordinates to the global coordinates

            det_jac = np.linalg.det(jac) #calculate the determinate of the jacobian

            #perform vector multiplication involving shape functions or derivatives (evaluated at integration points), multiplied by weight and by the det(J)
            #sum with the previous multiplication
            Mloc  = Mloc  + np.matmul(N_l.transpose(),N_l)*weight[i]*weight[j]*det_jac
            Kloc  = Kloc  + np.matmul(dN_g.transpose(),np.matmul(D,dN_g))*weight[i]*weight[j]*det_jac
            Floc  = Floc  + source*N_l.transpose()*weight[i]*weight[j]*det_jac




    
    #########################################
    
    Lloc = Mloc/dt + Kloc
    Rloc = Mloc/dt 

    n_now   =   g_num[:,iel]           #which nodes are in the current element?

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
        b[i]= bc_val
    #SOLVER

    T   = np.linalg.solve(LG,b)   #new temperature

    time = time + dt

    #PLOTTING

    #plt.plot(x_ana, T_ana, '-', color='red')


xv = xv.reshape(n_nodes_x,n_nodes_y)
yv = yv.reshape(n_nodes_x,n_nodes_y)
T = T.reshape(n_nodes_x,n_nodes_y)

print(xv)
print(yv)
plt.figure()
plt.contourf(xv,yv,T)
plt.colorbar()
plt.show()
'''


    
