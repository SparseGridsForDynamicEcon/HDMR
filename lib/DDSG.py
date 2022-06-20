## DDSG Class for Function approxmation 
""" 
This code supplements the work of Eftekhari and  Scheidegger titled High-Dimensional Dynamic Stochastic Model Representation, in SIAM Journal on Scientific
Computing (SISC), 2022 which introduces DDSG, a highly scalable function approximation technique. The lightweight (MPI parallel) Python implementation
presented here is intended showcase its applicability practical.

The DDSG technique is grid based function approximation method which combines a variant of Dimensional Decomposition known as High-Dimension Model 
Representation (HDMR), and adaptive Sparse Grid. The combined approach allows for a highly performant and scalable gird base function approximation method 
which can scale to high-dimensions.  

Please cite this paper in your publications if it helps your research:

Eftekhari, Aryan and Scheidegger, Simon (2022); High-Dimensional Dynamic Stochastic Model Representation
@article{Eftekhari_Scheidegger_2022,
  title={High-Dimensional Dynamic Stochastic Model Representation},
  author={Eftekhari, Aryan and Scheidegger, Simon},
  journal={forthcoming in the SIAM Journal on Scientific Computing - Available at SSRN 3603294},
  year={2022}
}
"""

import Tasmanian
from mpi4py import MPI
import numpy as np
from itertools import combinations
import os
import uuid
import time
import dill

class DDSG:

    def __init__(self,folder_name:str=None):
        """Constructor of DDSG class

        Args:
            folder_name (str, optional): Folder name to load a ddsg dump. Defaults to None.
        """

        if folder_name==None:
            pass
        else:
            with open(folder_name+'/ddsg', 'rb') as handle:
                obj=dill.load(handle)

            self.__dict__ = obj.__dict__

            self.proc_comm = MPI.COMM_WORLD.Dup()
            self.proc_size = obj.proc_comm.Get_size()
            self.proc_rank = obj.proc_comm.Get_rank()

            for u  in self.sg_obj:
                if len(u)!=0:
                    self.sg_obj[u] = Tasmanian.SparseGrid()
                    self.sg_obj[u].read(self._folder_sg_path(folder_name,u))

    def init(self,f_orical:object, d:int, m:int=1):
        """Initilize DDSG object

        Args:
            f_orical (object): Scalar valued function.
            d (int): Dimension of grid.
            m (int,optional): Degrees of freedom (Defual =1).           
        """

        assert d>1, 'd must be greater than 1.'

        np.random.seed(1)
        self.zero = np.finfo(np.float64).eps * 2 

        # f_orical : R^d -> R^k , with 
        # x0 is the anchor points
        # S = {1,2,...,d}
        self.d        = d
        self.m        = m
        self.f_orical = f_orical
        self.S        = range(1,self.d+1)
        self.x0       = None

        # Flags
        self._flag_no_hdmr           = False
        self._flag_grid_set          = False
        self._flag_decomposition_set = False
        self._flag_build_complete    = False
      
        # Variable defintions see _reset_data_structures()                    
        self.hdmr_obj    = None
        self.sg_obj      = None 
        self.settings    = {}
        self.X_sample    = None
        
        # Other Information
        self.num_grid_points = None           
        self.num_func        = None  

        # MPI world communicator
        self.proc_comm = MPI.COMM_WORLD.Dup()
        self.proc_size = self.proc_comm.Get_size()
        self.proc_rank = self.proc_comm.Get_rank()
        
        # MPI groups
        self.proc_group_number = None
        self.proc_group_comm   = None
        self.proc_group_size   = None
        self.proc_group_rank   = None
        self.proc_group_count  = None 
                
        # Every rank has the same id
        self.id = uuid.uuid4().hex
        self.id = self.proc_comm.bcast(self.id, root=0)
        
        # Temporary folder name
        self.temp_folder  = os.getcwd()+"/__TEMP_DDSG__"+str(self.id)

        # Second layer of parallelization in SG
        self.sg_prl=False
 
        self._reset_data_structures()
                  
    def set_grid(self,l_max:int,domain:np.array,eps_sg:float,grid_order:int=1,grid_rule:str='localp'):
        """Set adaptive SG paramters, based on TASMANIAN. See https://github.com/ORNL/TASMANIAN for further details.

        Args:
            domain (np.array): The domain to be approximated, a (d,2) matrix.
            l_max (int): Maximum refinment level of SG.
            eps_sg (float): Adaptive SG tolerance.
            grid_order (int, optional): An integer no smaller than -1 indicating the largest polynomial order of the basis functions (see TASMANIAN grid_order). Defaults to 1.
            grid_rule (str, optional): Local polynomial rules (see TASMANIAN grid_rule). Defaults to 'localp'.
        """

        assert domain.shape==(self.d,2), 'domain must have shape (d,2).'
        
        for i in range(self.d):
            assert domain[i,1] > domain[i,0], 'domain limits bust be such that domain[i,1]>domain[i,0] for all i'
            
        assert l_max>0 ,            'l_max must be great than 0.'
        assert eps_sg>=0 ,          'eps_sg must be great than or equal to 0. '        
        assert grid_order >=-1,     'grid_order must b greater or equal to -1.'
        
        self.settings['domain']     = domain
        self.settings['l_max']      = l_max
        self.settings['eps_sg']     = max(eps_sg,self.zero)
        self.settings['grid_order'] = grid_order
        self.settings['grid_rule']  = grid_rule
        self.settings['l_start']    = min(1,l_max)
        
        self._flag_grid_set         = True
        self._flag_build_complete   = False
        self._flag_no_hdmr          = True 

    def set_decomposition(self,x0:np.array,k_max:int,eps_rho:float=0,eps_eta:float=0,N_samples:int=1000): 
        """Set DD parmeters. If not set, DDSG acts as MPI parallel wrapper for TASMANIAN.

        Args:
            x0 (np.array): Achnor point with dimension d.
            k_max (int): Maximum expansion order.
            eps_rho (float, optional): Convergence criterion tolerance. Defaults to 0.
            eps_eta (float, optional): Active dimension selection tolerance. Defaults to 0.
            N_samples (int, optional): Number of samples for approximate quadrature. Defaults to 1000.
        """

        # set achor point
        

        assert x0.shape ==(1,self.d) ,'x0, must be of size 1xd, but it is'+str(x0.shape)+'.' 
        assert k_max> 0              ,'k_max must be greater than 0.'  
        assert eps_rho>=0            ,'eps_rho must be great than or equal to 0.'
        assert eps_eta>=0            ,'eps_eta must be great than or equal to 0.'  
        assert N_samples>1           ,'N_samples must be great than 1.' 
    
        # set anchor point
        self.x0 = x0

        # Global sync of random sample need in HDMR
        self.X_sample = np.empty(shape=(N_samples,self.d))
        if self.proc_rank==0:
            self.X_sample = np.random.uniform(low=self.settings['domain'][:,0], high=self.settings['domain'][:,1], size=(N_samples,self.d)) 
        self.X_sample = self.proc_comm.bcast(self.X_sample,root=0)  

        # level before which we do not adapativity
        self.settings['k_max']    = min(k_max,self.d)
        self.settings['eps_rho']  = max(eps_rho,self.zero)
        self.settings['eps_eta']  = max(eps_eta,self.zero)
               
        self._flag_decomposition_set    = True
        self._flag_build_complete       = False
        self._flag_no_hdmr              = False 
        
    def build(self,verbose:int=0):
        """Build the interpolant.

        Args:
            verbose (int, optional): Display runtime information 0,1,2,3,4 and 99. Defaults to 0.
        """
        # error check etc...
        assert self._flag_grid_set , 'grid is not set, use .set_grid(...)'
        if not self._flag_no_hdmr: # Using hdmr
            assert self._flag_decomposition_set, 'decomposition is not set, use .set_decomposition(...)'
            
        # Load SG setting ... they are need for both methods
        l_max   = self.settings['l_max']
        l_start = self.settings['l_start']
        eps_sg  = self.settings['eps_sg']
        grid_order  = self.settings['grid_order']
        grid_rule  = self.settings['grid_rule']

        # reset all data structures
        self._reset_data_structures()

        # Compute pure SG
        if self._flag_no_hdmr:
        
            t = time.time()
            
            if verbose>0 and self.proc_rank==0: print('### SG (sg_prl=%s): d=%d m=%d l_max=%d ϵ_sg=%0.2e  '%(self.sg_prl,self.d,self.m,l_max,eps_sg))
            
            # reallocate resrouces
            self._set_group_comm(num_tasks=1,verbose=verbose)

            # generate Sparse Grid over the full dimensional space
            u=(-1,)
            active_inx=list(range(0,self.d))
            sg = self._make_sparse_grid(l_start,l_max,eps_sg,active_inx,grid_order,grid_rule,verbose)
            self.num_grid_points   += sg.getNumPoints()
            self.num_func          += 1
            self.sg_obj[u]          = sg
            
            if verbose>1 and self.proc_rank==0: print('- Building (Adaptive) Sparse Grid (%0.2e sec.)'%(time.time()-t))
            
            self._flag_build_complete=True

        # compute HDMR+SG
        else: 
            
            # Load DDSG setting ... they dont exist if we have  ASG
            k_max   = self.settings['k_max']
            eps_rho = self.settings['eps_rho']
            eps_eta = self.settings['eps_eta']
            
            if verbose>0 and self.proc_rank==0: print('### DDSG (sg_prl=%s): d=%d m=%d k_max=%d l_max=%d ϵ_sg=%.2e ϵ_ρ=%.2e ϵ_η=%.2e '%(self.sg_prl,self.d,self.m,k_max,l_max,eps_sg,eps_rho,eps_eta))

            # zeroth expansion order  
            t = time.time()      
            u       = ()
            
            f0 = self.f_orical(self.x0)
            if np.isscalar(f0):
                f0 = np.array([f0])
            else:
                f0 = f0.reshape(self.m)
            
            f0_quad = f0 

            self.sg_obj[u]                 = f0
            self.hdmr_obj['cfunc_quad'][u] = f0_quad

            self.hdmr_obj['eta'][u] = np.linalg.norm(f0_quad)
            self.hdmr_obj['quad_k'].append(np.linalg.norm(f0_quad))

            self.num_grid_points += 1
            self.num_func        += 1

            if verbose>1 and self.proc_rank==0: print('- Expansion k=0 of %d continue (%0.2e sec.)'% (k_max,time.time()-t))

            # make the temporary __TEMP_DDSG__  folder ...
            self._folder_make(self.temp_folder)

            # All higher order expansion orders
            for k in range(1,k_max+1):
            
                t = time.time()
                        
                # currrent full index set
                U_k = list(combinations(self.S,k)) 

                # reallocate compute resources
                self._set_group_comm(num_tasks=len(U_k),verbose=verbose)
                        
                cfunc_quad_temp=[]
                eta_temp=[]
                u_temp=[]
                z_temp=[]
                    
                for i, u in enumerate(U_k):

                    # round-robbin work allocation on COMM_GROUP
                    if self.proc_group_number == i%self.proc_group_count:

                        # SG approximation of the subsbase spanned by the basis indicies u
                        # if u_temp_loc is None thatn the SG is dropped 
                        [sg_temp_loc,cfunc_quad_temp_loc,eta_temp_loc,u_temp_loc,z_temp_loc]=self._compute_sg_func(u=u,verbose=verbose)

                        if self.proc_group_rank==0:
                            if u_temp_loc is not None:
                                # write the SG to file
                                # we cannot share this object (Tasmanian is a shared library written in c++)
                                sg_temp_loc.write(self._folder_sg_path(self.temp_folder,u))                            
                                
                                cfunc_quad_temp.append(cfunc_quad_temp_loc)
                                eta_temp.append(eta_temp_loc)
                                u_temp.append(u_temp_loc)
                            else:    
                                z_temp.append(z_temp_loc)


                # global sync of relevant data, implicit barrier 
                # data is index by the rank of the process
                # note SG is not shared here, but rather read/loaded from disk
                block_container = []
                block_container.append(cfunc_quad_temp)
                block_container.append(eta_temp)
                block_container.append(u_temp)
                block_container.append(z_temp)
                
                block_container_nested=self.proc_comm.allgather(block_container)

                # remove the nested structures, but dont conver to array
                cfunc_quad_temp=[]
                eta_temp=[]
                u_temp=[]
                z_temp=[]

                for block_container in block_container_nested:
                    cfunc_quad_temp += block_container[0]
                    eta_temp        += block_container[1]
                    u_temp          += block_container[2]
                    z_temp          += block_container[3]

                # assemble accepted values
                cfunc_quad_sum_k=0.0
                for i,u in enumerate(u_temp):
 
                    # load the SG from disk than delete the file after
                    self.sg_obj[u] = Tasmanian.SparseGrid()
                    self.sg_obj[u].read(self._folder_sg_path(self.temp_folder,u))
                    
                    self.hdmr_obj['cfunc_quad'][u] = cfunc_quad_temp[i]
                        
                    cfunc_quad_sum_k              += cfunc_quad_temp[i]

                    self.hdmr_obj['eta'][u]        = eta_temp[i]
                    self.num_grid_points          += self.sg_obj[u].getNumPoints()
                    self.num_func                 += 1

                # update reject set
                self.hdmr_obj['Z_list']+=z_temp
                
                # current approximate quadrature at expansion order k
                self.hdmr_obj['quad_k'].append( self.hdmr_obj['quad_k'][k-1] + cfunc_quad_sum_k )

                # expanion criterion - | quad_k - quad_(k-1) | /  | quad_k - quad_(k-1) |
                # small number self.zero to previent numerical issues
                rho = np.linalg.norm( cfunc_quad_sum_k )/ (self.zero+np.linalg.norm( self.hdmr_obj['quad_k'][k-1] ) )
                self.hdmr_obj['rho'].append(rho)

                if rho<=eps_rho:
                    if verbose>1 and self.proc_rank==0: print('- Expansion k=%d of %d truncated ρ=%0.2e≤%0.2e (%0.2e sec.)'%(k,k_max,rho,eps_rho,time.time()-t))
                    break
                elif k==k_max :
                    if verbose>1 and self.proc_rank==0: print('- Expansion k=%d of %d end ρ=%0.2e (%0.2e sec.)'%(k,k_max,rho,time.time()-t))
                    break
                else:
                    if verbose>1 and self.proc_rank==0: print('- Expansion k=%d of %d continue ρ=%0.2e>%0.2e (%0.2e sec.)'%(k,k_max,rho,eps_rho,time.time()-t))
                
            # compute coefficent of sg 
            lookup_index = list(self.sg_obj.keys())
            self.hdmr_obj['vec_coeff'] = np.zeros(len(lookup_index))

            for inx,u in enumerate(lookup_index):
                len_u = len(u)
                # v \subseteq u for r={|u|,|u|-1,...,0} 
                for k in range(0,len(u)+1):
                    V = list(combinations(u,k))
                    len_u_less_len_v = len_u - k
                    for v in V:
                        if v in lookup_index:
                            self.hdmr_obj['vec_coeff'][lookup_index.index(v)] += np.power(-1,len_u_less_len_v)

            # decomposition Complete ...
            # remove __TEMP_DDSG__  folder ...
            self._folder_remove(self.temp_folder)
            self._flag_build_complete=True
      
    def eval(self,X:np.array)->np.array:
        """Evaluate the interpolant at the point(s) X, with every row being coordinate. 

        Args:
            X (np.array): An N by d matrix, where N is the number of points and d is the dimension input vector.

        Returns:
            np.array: An array of interpolant values. 
        """
        
        X=np.array(X)
        [N,d] = X.shape

        # Error checks
        assert self._flag_build_complete , 'Interplant has not be build, use .build'
        assert d == self.d, "Input dimension not correct."

        if self._flag_no_hdmr : # if pure SG
            u=(-1,)
            Y = self.sg_obj[u].evaluateBatch(X)

        else: # if hdmr with sg
            lookup_index = list(self.sg_obj.keys())
            Y = np.zeros(shape=(N,self.m))

            for i,u in enumerate(lookup_index):
    
                if self.hdmr_obj['vec_coeff'][i]==0:
                    continue
                k = len(u)

                if(k==0):
                    Y += self.sg_obj[u] * self.hdmr_obj['vec_coeff'][i]
                else:
                    #extract partial
                    active_inx = np.array(u)-1
                    X_partial = X[:,active_inx].reshape((N,k))

                    # evaluate specific points    
                    Y += self.sg_obj[u].evaluateBatch(X_partial) * self.hdmr_obj['vec_coeff'][i]
        return Y

    def get_points_values(self)->list:
        """Returns the grid points and function values used in the approximation.

        Returns:
            list[np.array,np.array]: A list of grid points and corresponding values.
        """

        #Error checks
        assert self._flag_build_complete , 'Interplant has not be fit, use .fit'
        
        X_values = np.empty(0)
        Y_values = np.empty(0)

        if self._flag_no_hdmr:
            u=(-1,)
            X_values = self.sg_obj[u].getPoints()
            Y_values = self.sg_obj[u].getLoadedValues()
        else:
            lookup_index = list(self.sg_obj.keys())
            
            #for key, obj in self.sg_obj.items():
            for u in lookup_index:

                if len(u)==0:
                    X_values = self.x0
                    Y_values = self.sg_obj[u] 
                else:
                    active_inx = np.array(u)-1
                    # get poinst from SG and mask them x0
                    num_grid_points      = self.sg_obj[u].getNumPoints()
                    X_mask               = np.tile(self.x0,(num_grid_points,1))
                    X_mask[:,active_inx] = self.sg_obj[u].getPoints()
                    X_values             = np.vstack((X_values,X_mask))

                    # get points from SG and mask them x0
                    Y_values = np.vstack((Y_values,self.sg_obj[u].getLoadedValues()))

        return [X_values,Y_values]
  
    def benchmark(self,N:int,verbose:int=0)->list:
        """Basic unit test for accurecy and runtime.

        Args:
            N (int): Number of samples used in the test.
            verbose (int, optional): Display runtime information 0,1,2,3,4 and 99. Defaults to 0.

        Returns:
            list[float,float,int,int,float,float,float]: Results of benchmark: Max error, L2 error, number of grid points, number of component functions, the time needed to build the approximation, average time of interpolant evaluation, average time for calling the oracle function.
        """

        X = np.empty(shape=(N,self.d))
        if self.proc_rank==0:
            X = np.random.uniform(low=self.settings['domain'][:,0], high=self.settings['domain'][:,1], size=(N,self.d))  
        X = self.proc_comm.bcast(X,root=0) 
        
        t_orical_single   = -time.time()
        Y_orical          = self.f_orical(X)
        t_orical_single  += time.time()
        t_orical_single   = t_orical_single/N
        
        t_build   = -time.time()
        self.build(verbose=verbose) 
        t_build  += time.time()
        
        t_eval   = -time.time()
        Y_interp = self.eval(X).reshape(Y_orical.shape)
        t_eval  += time.time()
 
        err_diff = (Y_orical -Y_interp)
        err_l2   = np.linalg.norm(err_diff) / np.linalg.norm(Y_orical) 
        err_max  = np.max(abs(err_diff))

        # The runtime differ slightly between each node, we take the average
        t_orical_single = self.proc_comm.allreduce(t_orical_single,MPI.SUM)/self.proc_size
        t_build         = self.proc_comm.allreduce(t_build,MPI.SUM)/self.proc_size
        t_eval          = self.proc_comm.allreduce(t_eval,MPI.SUM)/self.proc_size

        return [float(err_max),float(err_l2),int(self.num_grid_points),int(self.num_func),float(t_build),float(t_eval),float(t_orical_single)]

    def dump(self,folder_name:str,replace=False):
        """Dump the DDSG object to file.

        Args:
            folder_name (str): Name of folder to store the DDSG dump files.
            replace (bool, optional): Overwrite folder if exists. Defaults to False.
        """

        self.proc_comm.barrier()
        if self.proc_rank==0:

            if replace==True:
                self._folder_remove(folder_name)
                self._folder_make(folder_name)
            else:
                self._folder_make(folder_name)
    
            self.proc_comm = None
            self.proc_size = None
            self.proc_rank = None
            self.proc_group_comm = None

            for u in self.sg_obj:
                if len(u)!=0:
                    self.sg_obj[u].write(self._folder_sg_path(folder_name,u))
                    self.sg_obj[u]=None
                
            with open(folder_name+'/'+'ddsg', 'wb') as handle:
                dill.dump(self, handle, protocol=dill.HIGHEST_PROTOCOL)

            self.proc_comm = MPI.COMM_WORLD.Dup()
            self.proc_size = self.proc_comm.Get_size()
            self.proc_rank = self.proc_comm.Get_rank()
            
            for u in self.sg_obj:
                if len(u)!=0:
                    self.sg_obj[u] = Tasmanian.SparseGrid()
                    self.sg_obj[u].read(self._folder_sg_path(folder_name,u))

    def _set_group_comm(self,num_tasks:int,verbose:int==0):
        """Allocate the MPI processes.

        Args:
            num_tasks (int): Number of tasks.
            verbose (int, optional): Display runtime information 0,1,2,3,4 and 99. Defaults to 0.
        """

        self.proc_comm.barrier()

        # full model of MPI_COMM
        tasks            = np.array(range(0,num_tasks))
        proc_ranks       = np.array(range(0,self.proc_size)).astype(int)
        group_size_floor = max(1,np.floor(self.proc_size/num_tasks))
        group_number     = ((proc_ranks / group_size_floor)%num_tasks ).astype(int)
        group_sizes      = np.bincount(group_number).astype(int)
        group_count      = len(group_sizes)

        if verbose==99 and self.proc_rank==0:
            print('@ MPI Configurations')
            print('@ Global Schema:')
            print('@    tasks                  =',tasks)
            print('@    proc_ranks             =',proc_ranks)
            print('@    group_size_floor       =',group_size_floor)
            print('@    group_number (color)   =',group_number)
            print('@    group_count            =',group_count )
            print('@    group_sizes            =',group_sizes)


        self.proc_group_number = group_number[self.proc_rank]
        self.proc_group_comm = self.proc_comm.Split(self.proc_group_number,self.proc_rank)
        self.proc_group_rank = self.proc_group_comm.Get_rank()
        self.proc_group_size = self.proc_group_comm.Get_size()
        self.proc_group_count = group_count

        if verbose==99:
            if self.proc_rank==0 : print('@ Local Allocation:')
            self.proc_comm.barrier()
            time.sleep(self.proc_rank+1)
            print('@    rank ',self.proc_rank,'/',self.proc_size,' maps to ','group_rank',self.proc_group_rank,'/',self.proc_group_size)
            self.proc_comm.barrier()

    def _folder_sg_path(self,folder_name:str,u:tuple )->str:
        """Make folder path for SG object.

        Args:
            folder_name (str): Folder name.
            u (tuple): Component index of SG.

        Returns:
            str: SG folder path.
        """
        return folder_name+'/'+','.join(map(str,u))+'.tasmanian'

    def _folder_make(self,folder_name:str):
        """Make folder.

        Args:
            folder_name (str): Path of the folder.
        """

        self.proc_comm.barrier()
        if self.proc_rank==0:
            assert not os.path.exists(folder_name), 'The folder exists!' 
            os.makedirs(folder_name)         
               
    def _folder_remove(self,folder_name:str):
        """Remove folder

        Args:
            folder_name (str): Path of the folder.
        """
        self.proc_comm.barrier()
        if self.proc_rank==0:
            if os.path.exists(folder_name):
        
                # delete files in folder
                for file_name in os.listdir(folder_name):
                    file_path = os.path.join(folder_name, file_name)
                    os.remove(file_path)
                                    
                #delete folder                   
                os.rmdir(folder_name)         

    def _reset_data_structures(self):
        """Rest the datastructures
        """
        
        self.sg_obj = {}                        #dic[tuple] ->Tasmanian
   
        self.hdmr_obj = {}
        self.hdmr_obj['eta']={}                 #dic[string][tuple] ->float
        self.hdmr_obj['rho']=[]                 #dic[string]        ->list:float
        self.hdmr_obj['cfunc_quad']={}          #dic[string][tuple] ->list:float
        self.hdmr_obj['quad_k']=[]              #dic[string]        ->list:float    
        self.hdmr_obj['vec_coeff']=np.empty(0)  #dic[string]        ->np.array()
        self.hdmr_obj['Z_list']= []             #dic[string]        ->list:tuple
    
        self.num_grid_points = 0           # int
        self.num_func        = 0           # int
                                                                
    def _make_sparse_grid(self,l_start:int,l_max:int,sg_tol:float,active_inx:np.array,grid_order:int,grid_rule:str,verbose)->object:
        """Wrapper for Tasmanian adaptive SG with MPI prallel function evaluations. 

        Args:
            l_start (int): Starting refiment level
            l_max (int): Maximum refiment level
            sg_tol (float): Adaptive SG tolerance.
            active_inx (np.array): Indicies for the DDSG component function (zero based index).
            grid_order (int):  An integer no smaller than -1 indicating the largest polynomial order of the basis functions (see TASMANIAN grid_order).
            grid_rule (str):  Local polynomial rules (see TASMANIAN grid_rule).
            verbose (int): Display runtime information 0,1,2,3,4 and 99.

        Returns:
            object: Tasmania SG interpolant.
        """

        assert len(active_inx)>0, 'Active index must at least one active index!' 
        
        # Construct the grid & set the domain
        # we start with refinment level l_start
        sg = Tasmanian.SparseGrid()
        sg.makeLocalPolynomialGrid(len(active_inx),self.m,l_start,grid_order,grid_rule)
        
        #set domain
        grid_domain  = self.settings['domain'][active_inx,:]
        sg.setDomainTransform(grid_domain)
        
        # loop through refimnet level upto and including l_max
        for l in range(l_start,l_max+1):

            # get grid points and setup for evaluations
            grid_points     = sg.getNeededPoints()
            num_grid_points = sg.getNumNeeded()

            # if no grid points than refinement has ended
            if num_grid_points<1:
                break

            # mask input in x
            if self._flag_no_hdmr :
                X_mask= grid_points
            else:
                X_mask               = np.tile(self.x0.flatten(),(num_grid_points,1))
                X_mask[:,active_inx] = grid_points

            if self.proc_group_size == 1 or self.sg_prl==False:
                f_val_buffer = self.f_orical(X_mask).reshape(num_grid_points,self.m)
            else:

                offset  = int(np.ceil(num_grid_points/self.proc_group_size))
                i_begin = max(0,self.proc_group_rank * offset) 
                i_end   = min(num_grid_points,(self.proc_group_rank+1) * offset)

                f_val_buffer_temp = self.f_orical(X_mask[i_begin:i_end,:])
                f_val_buffer_temp = self.proc_group_comm.allgather(f_val_buffer_temp)

                f_val_buffer      = np.concatenate(f_val_buffer_temp,axis=0).reshape(num_grid_points,-1)

            # load function values into the SG
            sg.loadNeededValues(f_val_buffer)

            # move to the next refinment level
            sg.setSurplusRefinement(sg_tol, -1, "classic")
            
            #if verbose>3 and self.proc_group_rank==0 and self._flag_no_hdmr: print('  ','[MPI Group Size' ,self.proc_group_size, '] SG: l=' ,l_start,'/',l_max,'#grid points=', num_grid_points,' (', np.round(time.time()-t,2),' sec.)')

        return sg

    def _compute_sg_func(self,u:tuple,verbose:bool)->list:
        """Generate the DDSG component function.

        Args:
            u (tuple): Indicies for the DDSG component function (1 based index).
            verbose (int): Display runtime information 0,1,2,3,4 and 99.

        Returns:
            list[object,float,float,tuple,tuple]: List of values include, the Tasmania SG interpolant, component function approximate quadrature, active dimension selection eta, component function index, and if rejected index. Note if active dimension selection eta is less than the threshold, then the component function is deemed ignorable, and thus, all values of this list are None, except for the rejected index, which would equal to the component function index. If the component function is not ignorable, only the rejected index is None.
        """
        
        t = time.time()
        
        sg_temp         = None
        cfunc_quad_temp = None
        eta_temp        = None
        u_temp          = None 
        z_temp          = None 

        l_max   = self.settings['l_max']
        l_start = self.settings['l_start'] 
        eps_sg  = self.settings['eps_sg']  
        eps_eta = self.settings['eps_eta'] 
        grid_order = self.settings['grid_order'] 
        grid_rule  = self.settings['grid_rule'] 

        # Current index u
        active_inx = np.array(u)-1

        # Check if index u should be ignore
        candidate_u = True
        for z in self.hdmr_obj['Z_list']:
            if set(u).issuperset(set(z)):
                if verbose>2 and self.proc_group_rank==0: print('  ','[MPI Group Size',self.proc_group_size,'] Index:',u,'[ignored',u,'⊃',z,']  (%0.2e sec.)'%(time.time()-t,))
                candidate_u = False
                break
            
        if candidate_u:

            # Generate Sparse Grid for len(u)-dimensional space
            sg = self._make_sparse_grid(l_start,l_max,eps_sg,active_inx,grid_order,grid_rule,verbose)

            q_cfv = 0.0
            for r_temp in range(0,len(u)):
                V_r = list(combinations(u,r_temp))
                for v in V_r:
                    if v in list(self.sg_obj.keys()):
                        q_cfv += self.hdmr_obj['cfunc_quad'][v]

            # sample the space
            # We can take the sg quadrature but ... simple sampling is enough
            sg_mc_quad = np.mean((sg.evaluateBatch(self.X_sample[:,active_inx])))*np.product(self.settings['domain'][:,1]-self.settings['domain'][:,0])
            q_cfu =sg_mc_quad - q_cfv
                            
            quad_norm_k_minus_1 = np.linalg.norm(self.hdmr_obj['quad_k'][-1] )
            quad_nrom_cfu = np.linalg.norm(q_cfu)
            
            # small number self.zero to previent numerical issues
            eta = (quad_nrom_cfu) / (self.zero+quad_norm_k_minus_1)

            if eta > eps_eta:
                sg_temp         = sg
                cfunc_quad_temp = q_cfu
                eta_temp        = eta
                u_temp          = u
                z_temp          =  None
                if verbose>2 and self.proc_group_rank==0:  print('  ','[MPI Group Size',self.proc_group_size,'] u=%s accepted η=%0.2e>%0.2e (%0.2e sec.)'%(u,eta,eps_eta,time.time()-t))
            else:
                sg_temp         = None
                cfunc_quad_temp = None
                eta_temp        = None
                u_temp          = None 
                z_temp          = u 
                if verbose>2 and self.proc_group_rank==0:  print('  ','[MPI Group Size',self.proc_group_size,'] u=%s ignored η=%0.2e≤%0.3e (%0.2e sec.)'%(u,eta,eps_eta,time.time()-t)) 
                
        return [sg_temp,cfunc_quad_temp,eta_temp,u_temp,z_temp]
