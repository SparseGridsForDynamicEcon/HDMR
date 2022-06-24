# add root into path 
import os
import sys

import time
import numpy as np
from tabulate import tabulate
from scipy import optimize
from IRBC import IRBC
from DDSG import DDSG

#input parameters
num_countries   = int(sys.argv[1])
l_max           = int(sys.argv[2])
k_max           = int(sys.argv[3])
iter_max        = int(sys.argv[4])

# IRBC model
model = IRBC(num_countries=num_countries, irbc_type='non-smooth')  
model.set_parameters()
model.set_integral_rule() 

def eq_condition(X):  
    global p_last
    [n,d]=X.shape
    result = np.empty(shape=(n,model.grid_dof))
    for i in range(0,n):
        state    = X[i,:]
        p_guess  = p_last.eval(X[i,:].reshape(1,-1))
        solution = optimize.root(fun=model.system_of_equations, x0=p_guess,tol=1e-10,args=(state,p_last), method='hybr') 
        result[i,:] = solution.x        
    return result

def eq_condition_init_guess(X):
    [n,d]=X.shape
    val = np.empty(shape=(n,model.grid_dof))
    for i in range(0,n): 
        val[i,0:model.num_countries]  = (model.k_min + model.k_max)/2
        val[i,model.num_countries]    = 1
        val[i,model.num_countries+1:] = -0.1

    return val

# main parameters
eps_sg   = 1e-3
l_max    = l_max
iter_max = iter_max

#domain of the grid
domain                          = np.zeros((model.grid_dim,2))
domain[0:model.num_countries,0] = model.k_min
domain[0:model.num_countries,1] = model.k_max
domain[model.num_countries:,0]  = model.a_min
domain[model.num_countries:,1]  = model.a_max

# hdmr anchor point ... is ignored if SG is used
x0=np.mean(domain,axis=1).reshape((1,domain.shape[0]))

# sample points for policy convergence/stagnation
X_sample = np.random.uniform(low=domain[:,0],high=domain[:,1],size=(1000,model.grid_dim))

# initial policy 'guessed' funciton ... corresponding to the "eq_condition_init_guess"
p_last = DDSG()
p_last.init(f_orical=eq_condition_init_guess,d=model.grid_dim,m=model.grid_dof) 
p_last.set_grid(domain=domain,l_max=l_max,eps_sg=eps_sg)

# if k_max is less than 1, we use SG
if k_max>0:
    p_last.set_decomposition(x0,k_max=k_max,eps_rho=1e-6,eps_eta=1e-6)

p_last.sg_prl=True
p_last.build(verbose=1)

if p_last.proc_rank==0:
    model.print_parameters()  

t_total       =[]
error_l2_mean =[]
grid_points   =[]

# time-iteration
for i in range(0,iter_max):
    
    t_total.append(-time.time())

    # construct new policy, i.e., p_next, using p_last
    p_next = DDSG()
    p_next.init(eq_condition,d=model.grid_dim,m=model.grid_dof)  
    p_next.set_grid(domain=domain,l_max=l_max,eps_sg=eps_sg)
    if k_max>0:
        p_next.set_decomposition(x0,k_max=k_max,eps_rho=1e-6,eps_eta=1e-6)

    p_next.sg_prl=True
    p_next.build(verbose=0)
    
    t_total[-1] += time.time()

    # evaluate the difference two incremental policies ... a measure of stagnation
    diff = p_next.eval(X_sample) - p_last.eval(X_sample)
    error_l2_mean.append(np.linalg.norm(diff.flatten())/diff.size) 
    grid_points.append(p_next.num_grid_points)

    if p_next.proc_rank==0:
        print('# iter:{:d} time(Sec):{:.2e}  error_l2:{:.2e}  gridpoints:{:.2e}'.format(i,t_total[-1],error_l2_mean[-1],grid_points[-1]) )

    # swap policy
    p_last = p_next

data=[]
headers = ['Cumulitive Rutme Time (Sec.)','Cumulitive Number of Grid Points']
data.append(['Result',np.sum(t_total),np.sum(grid_points)])

if p_next.proc_rank==0:
    print(tabulate(data,headers=headers))
