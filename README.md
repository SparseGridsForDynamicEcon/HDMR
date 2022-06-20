# High-Dimensional Dynamic Stochastic Model Representation
This code repository here supplements the work of Eftekhari and Scheidegger, titled _[High-Dimensional Dynamic Stochastic Model Representation](#publication)_ (Eftekhari and Scheidegger; SIAM SISC 2022), which introduces a highly scalable function approximation technique using Dimensional Decomposition and adaptive Sparse Grid (DDSG) to solve dynamic stochastic economic models. Concretely, the DDSG algorithm is embedded in a time-iteration algorithm to solve high-dimensional, nonlinear dynamic stochastic economic models. Furthermore, the introduced method can trivilly be extended to solve models with value function iteration. Note that our algorithm was originally developed in C++ and Fortran using hybrid parallelism (OpenMP and MPI); however, the MPI parallel Python implementation presented here is intended to be more practical, while still being decently performant in. Concretely:

* This repository provieds a versatile and generic method for approximating very high-dimensional functions.
* This repository provides a method that is applicable in computing recursive equilibria of nonlinear dynamic stochastic economic models with many state variables.
* The method has been demonstrated in the accompanied article that dynamic stochastic models with up to 300 state variables could be solved globally using DDSG.
* This repository aims to make our method easily accessible to the computational economics and finance community. 

![image](https://drive.google.com/uc?id=120KCXRvqwZHefPsUbjmkLex8pK5PMf1S)

This figure is a visual representation of one step of the time-iteration algorithm. We solve the first-order conditions (FOC) of the model for the state variable
$x_t$ 
in the updated policy function 
$\tilde{p}'$ 
(left), using the policy function from the previous time iteration step 
$\tilde{p}$ 
(right). In the following description, 
$\tilde{p}'$ 
and 
$\tilde{p}$ 
correspond to `p_next` and `p_last`, respectivily. These policy functions are approximated using DDSG.

## Libraries 
The primary libraries introduced in this repository are _DDSG_ for function approximation (used for both DDSG and adaptive SG) and _IRBC_ for IRBC model description.

To correctly import the libraries, their paths must be included in your system path. A simple method can be to hard code it using `sys.path.append('/path/to/this_repo/lib')`. Throughout this repository, we use the following code as the required path:

```
import os
import sys
# add root into path 
sys.path.append(os.path.dirname(pathlib.Path("__file__").absolute().parent.parent)) 
```
The path is encoded as two subdirectories back (corresponding to `parent.parent`) from the absolute path of the file being executed (corresponding to `pathlib.Path('__file__).absolute()`). For more details, refer to https://docs.python.org/3/tutorial/modules.html.


### lib/DDSG.py
The DDSG technique is a grid-based function approximation method that combines High-Dimension Model Representation, a variant of _Dimensional Decomposition_ (DD), and adaptive _Sparse Grid_ (SG). The combined approach enables a highly performant and scalable gird base function approximation method that can scale efficiently to high dimensions and utilize distributed memory architectures. This library is user-friendly and parallelized with MPI. The SG components of the algorithm use the [Tasmanian](https://tasmanian.ornl.gov) open-source SG library.

#### Usage
Using the DDSG requires the following steps:
1. Instantiate DDSG with the function to be approximated and the dimension of the domain, e.g., `ddsg=DDSG()` and `ddsg.init(f,d)`. If you are loading the object from a file, we can call `ddsg=DDSG(path_to/my_ddsg_obj)`.
2. Set the parameters for the adaptive SG (see documentation for details), e.g., `ddsg.set_grid(l_max=10,eps_sg=1e-6)`. 
3. Set the parameters for the DD decomposition (see documentation for details), e.g., `ddsg.set_decomposition(x0=np.ones(d)/2,k_max=1)`. If `set_decomposition()` is not invoked, the DDSG class works as a wrapper for the Tasmanian library, with built-in MPI parallelism.
4. Build the approximation, e.g., `ddsg.build(verbose=1)`.

If a specific grid point evaluation is computationally demanding, we can use MPI processes to evaluate the needed grid points in parallel by setting `ddsg.sg_prl=True` (by default `ddsg.sg_prl=False`). If the number of available MPI processes exceeds the number of component functions, the extra processes are assigned in the SG computation if `ddsg.sg_prl=True`.

```
#Example ddsg usage

from DDSG import DDSG
import numpy as np

d=10
f_example = lambda X: -15*np.sum( np.abs(X-4/11) ,axis=1)
x0=np.ones(shape=(1,d))/2

domain =np.zeros(shape=(d,2))
domain[:,1]=1

ddsg=DDSG()
ddsg.init(f_orical=f_example, d=d)
ddsg.set_grid(domain=domain,l_max=4,eps_sg=1e-6)
ddsg.set_decomposition(x0=x0,k_max=1)
ddsg.build(verbose=1)

x=np.random.uniform(low=0, high=1,size=x0.shape) 
print('val_est=',ddsg.eval(x))
print('val_true=',f_example(x))
```

#### Parallel Execuation
Parallel execution follows straightforwardly using `mpirun`. It is recommended that `--bind-to core` option be used to ensure that the MPI processes are bound to physical cores.
```
mpirun -np 4 --bind-to core python3 file_to_run.py
```
_Note that for parallelization within the sparse grid, the option `ddsg.sg_prl=True` must be set._


### lib/IRBC.py
The International Real Business Cycle (IRBC) library supports two models: _smooth_ and _non-smooth_. We refer to an IRBC model as smooth if there are no kinks in the policies and non-smooth if there are non-differentiabilities in the latter functions. The models are simple to describe, have a unique solution, and their dimensionality can be meaningfully scaled up. As such, these models are used to test various solution strategies for large-scale dynamic stochastic economic models. This model trait allows us to focus on the computational problems of dealing with high-dimensional state spaces. 

_The models are implemented and parameterized (by default) as per the article by Brumm and Scheidegger titled [Using Adaptive Sparse Grids to Solve High-Dimensional Dynamic Models]([https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3979412](https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA12216))(2017). For further details, we refer the reader to that article._

#### Usage
Using the IRBC model requires the following steps:
1. Instatitate the IRBC model with the number of countries (say 2) and type of model (smooth or non-smooth), i.e., `model=IRBC(num_countries=2,irbc_type='non-smooth')`.
2. Set the parameters of the model, i.e.,`.set_parameters()`. Note that all parameters are set by default to those found in the aforementioned publications.
3. Set the default integral rules, i.e., `.set_integral_rule()`. Note that at the moment, only 'monomials_power' has been implemented, and this is selected by default.
4. (optional) For confirmation, we can print the parameters of the model using `print_parameters()` method.

```
from IRBC import IRBC
import numpy as np

model = IRBC(num_countries=2, irbc_type='non-smooth') 
model.set_parameters()
model.set_integral_rule() 
mode.print_parameters()
```

#### Integration with DDSG
The IRBC library is intended to be used with the DDSG library. The main method of the IRBC object is the `.system_of_equations(x, state,p_last)`, which is the value of the residual of the first-order-conditions optimality conditions of the model. In particular, `x` is the policy to be solved at the state denoted by `state`, and `grid` is the last best estimate of the policy function (i.e., a DDSG approximation of the current policy function). The zeros of this system of (non-linear) equations must be solved for all discrete states (i.e., grid points) in the state space.

The following example shows how we can incorporate the DDSG into the IRBC mode. In this case, `p_rand` is a function with appropriate bounds (all set to that of the capital range k_min to k_max). From here on, we generate a DDSG approximation of the random function, which we treat as the estimate of some policy function called `p_last`. Finally, we can compute the residual of first-order optimality conditions if the optimal policy `x=p_guess` at state `state=state` is given the current DDSG policy approximation `grid=p_rand`.

```
from DDSG import DDSG
from IRBC import IRBC
import numpy as np

model = IRBC(num_countries=2, irbc_type='smooth') 
model.set_parameters()
model.set_integral_rule() 

p_rand = lambda X: np.random.uniform(low=model.k_min, high=model.k_max, size=(X.shape[0],model.grid_dof))

domain = np.zeros((model.grid_dim,2))
domain[0:model.num_countries,0] = model.k_min
domain[0:model.num_countries,1] = model.k_max
domain[model.num_countries:,0] = model.a_min
domain[model.num_countries:,1] = model.a_max

x0 = np.mean(domain,axis=1).reshape((1,domain.shape[0]))

p_last = DDSG()
p_last.init(f_orical=p_rand,d=model.grid_dim,m=model.grid_dof) 
p_last.set_grid(domain=domain,l_max=4,eps_sg=1e-3)
p_last.set_decomposition(x0,k_max=1,eps_rho=1e-3,eps_eta=1e-3)
p_last.build(verbose=0)

state = np.array([1.0,1.0,0.0,0.0])
p_guess = np.array([1.0,1.0,1.0,-0.1])
foc_residual = model.system_of_equations(x=p_guess,state=state,grid=p_rand)
print('foc_residual=',foc_residual)

```

## Examples: 
The first example provided outlines analytical test cases for the general DDSG function approximation technique, whereas the second example focuses on using DDSG as part of the IRBC model solution. Both examples include a tutorial base Jupiter notebook that is intended to be pedagogical. Furthermore, we provided standalone Python scripts for each of the examples to highlight the performance and scalability of the introduced computational methods.

### examples/analytical
The analytical examples provided for DDSG cover the following topics:
1. How to use the DDSG library.
2. Fundamentals of SG and DDSG approximation.
3. Seperablity of functions (i.e., decomposition).
4. Performance and the effect of cure-of-dimensionality.
5. Scalability and execution on distributed memory architectures (parallel execution). _The standalone python script `examples/analytical/unit_test.py` is used in these tests._

[![Generic badge](https://img.shields.io/badge/jupyter%20nbviewer-DDSG-green)](https://nbviewer.jupyter.org/github/https://github.com/SparseGridsForDynamicEcon/py-HDMR/tree/main/examples/analytical/tutorial.ipynb)


### examples/irbc
The IRBC examples provided here cover the following topics:
1. How to use the IRBC library.
2. Incorporating the DDSG library within the IRBC mode.
3. Using the time-iteration method (along with the DDSG library) to solve the optimal policy of the IRBC mode.
4. Using the DDSG library to run both SG and DDSG approximation of the optimal policy function. 
5. Computing metrics such as stagnation and simulation error of the policy function. 
6. Scalability and performance of using DDSG in place of just SG (parallel execution). _The standalone python script `examples/irbc/unit_test.py` is used in these tests._

[![Generic badge](https://img.shields.io/badge/jupyter%20nbviewer-IRBC-green)](https://nbviewer.jupyter.org/github/https://github.com/SparseGridsForDynamicEcon/py-HDMR/tree/main/examples/irbc/tutorial.ipynb)


## Publication

Please cite [High-Dimensional Dynamic Stochastic Model Representation, A. Eftekhari, S. Scheidegger, SIAM Journal on Scientific Computing (SISC), 2022](https://epubs.siam.org/doi/10.1137/21M1392231) in your publications if it helps your research:
```
@article{doi:10.1137/21M1392231,
 author = {Eftekhari, Aryan and Scheidegger, Simon},
 title = {High-Dimensional Dynamic Stochastic Model Representation},
 journal = {SIAM Journal on Scientific Computing},
 volume = {44},
 number = {3},
 pages = {C210-C236},
 year = {2022},
 doi = {10.1137/21M1392231}
}
```
See [here](https://arxiv.org/pdf/2202.06555.pdf) for an archived version of the article. 


### Authors
* [Aryan Eftekhari](https://scholar.google.com/citations?user=GiugKBsAAAAJ&hl=en) (Department of Economics, University of Lausanne)
* [Simon Scheidegger](https://sites.google.com/site/simonscheidegger/) (Department of Economics, University of Lausanne)


### Other Relate Resreach 
* [Using Adaptive Sparse Grids to Solve High-Dimensional Dynamic Models; Brumm & Scheidegger (2017)](https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA12216)).
* [Sparse Grids for Dynamic Economic Models; Brumm et al. (2022)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3979412)(2021)


## Prerequisites / Installation
_SG library (required by DDSG)_
```shell
$ pip3 install Tasmanian
```
For further information on alternative installation procedures, see https://tasmanian.ornl.gov/documentation/md_Doxygen_Installation.html.

_Optimization and general numerics_
```
$ pip3 install scipy 
$ pip3 install numpy 
```

_Parallelization_
```
$ pip3 install mpi4py 
```

_Visualization and tabulation_
```
$ pip3 install matplotlib 
$ pip3 install tabulate
```

## Support
This work is generously supported by grants from the [Swiss National Science Foundation](https://www.snf.ch) under project IDs “New methods for asset pricing with frictions”, "Can economic policy mitigate climate change", and the [Enterprise for Society (E4S)](https://e4s.center).
