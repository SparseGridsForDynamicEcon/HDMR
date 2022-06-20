# add root into path 
import os
import sys
import pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path("__file__").absolute().parent.parent)) 

import time
import numpy as np
from tabulate import tabulate
from lib.DDSG import DDSG 

# get parameters from command line
d       = int(sys.argv[1])
l_max   = int(sys.argv[2])
k_max   = int(sys.argv[3])

domain =np.zeros(shape=(d,2))
domain[:,1]=1
    
# assuming a computaionaly expensive function call
def f_example_heavey(X):
    n = X.shape[0]
    val = np.zeros(n);

    for i in range(0,n):
        val[i]=-15*np.sum( np.abs(X[i,:]-4/11))
        time.sleep(0.01)
        
    return val 

x0=np.ones(shape=(1,d))/2.0
ddsg = DDSG()
ddsg.init(f_example_heavey,d)
ddsg.set_grid(domain=domain,l_max=l_max,eps_sg=1e-6)
ddsg.sg_prl=True

if k_max>0:
    ddsg.set_decomposition(x0,k_max=k_max,eps_rho=1e-6,eps_eta=1e-6)

[err_max,err_l2,num_grid_points,num_func,t_build,t_eval,t_orical]= ddsg.benchmark(N=1000,verbose=4)

if(ddsg.proc_rank==0):
    headers = ['Error-Max','Error-L2','#Grid Points','#Comp. Func.','Time-Build','Time-Eval','Time-Orical']
    print('\n### Benchmark Results')
    print(tabulate([[err_max,err_l2,num_grid_points,num_func,t_build,t_eval,t_orical]] ,headers=headers))