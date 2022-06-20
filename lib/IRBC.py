"""
This Python code accompanies the review article by Brumm, Krause, Schaab, & Scheidegger (2021)
and corresponds to the International Real Business Cycle (IRBC) model. See paper for further details.

This class is a reimplentation of the model outlined in https://github.com/SparseGridsForDynamicEcon/SparseGrids_in_econ_handbook

"""


import numpy as np
from tabulate import tabulate

class IRBC:
    
    def __init__(self, num_countries:int, irbc_type:str):
        """Constructor of IRBC class

        Args:
            num_countries (int): Number of countries in the model.
            irbc_type (str): Type of IRBC model (smooth or non-smooth)
        """
        
        ## Economic parameters
        #assert irbc_type=='smooth'
         
        # Intertemporal elasticity of substitution
        self.gamma = None
        self.ies_a = None
        self.ies_b = None
        
        # Discount factor
        self.beta = None
        # Capital share of income
        self.zeta = None
        # Depreciation rate
        self.delta = None
        # Persistence of TFP shocks
        self.rho_Z = None
        # Standard deviation of TFP shocks
        self.sig_E = None
        # Intensity of capital adjustment costs
        self.kappa = None
        # Aggregate productivity
        self.A_tfp = None
        # Welfare weight
        self.pareto = None
        # Lower bound for capital
        self.k_min = None
        # Upper bound for capital
        self.k_max = None
        
        # Lower bound for TFP
        self.a_min = None
        # Upper bound for TFP
        self.a_max = None

        # Number of countries
        self.num_countries = num_countries
        
        # Number of shocks (Country-specific shocks + aggregate shock)
        self.num_shocks = self.num_countries+1
        
        # Number of policies (nCountries+1 for smooth IRBC, nCountries*2+1 for nonsmooth)
        self.irbc_type=irbc_type
        if self.irbc_type=='non-smooth':
            self.num_policies = self.num_countries*2+1

            
        elif self.irbc_type=='smooth':
            self.num_policies = self.num_countries+1
       
        self.grid_dim = self.num_countries*2
        self.grid_dof = self.num_policies
        
        self.flag_set_param    = False
        self.flag_set_integral = False
        
    def set_parameters(self,ies_a:float=0.25,ies_b:float=1,beta:float=0.99,zeta:float=0.36,delta:float=0.01,rho_Z:float=0.95,sig_E:float=0.01,kappa:float=0.5,k_min:float=0.8,k_max:float=1.2):
        """Set IRBC paramters. Note the defualt settings follows the model described in https://github.com/SparseGridsForDynamicEcon/SparseGrids_in_econ_handbook.

        Args:
            ies_a (float, optional): Intertemporal elasticity of substitution upper. Defaults to 0.25.
            ies_b (float, optional): Intertemporal elasticity of substitution lower. Defaults to 1.
            beta (float, optional): Discount factor. Defaults to 0.99.
            zeta (float, optional): Capital share of income. Defaults to 0.36.
            delta (float, optional): Depreciation rate. Defaults to 0.01.
            rho_Z (float, optional): Standard deviation of TFP shocks. Defaults to 0.95.
            sig_E (float, optional): Standard deviation of TFP shocks. Defaults to 0.01.
            kappa (float, optional): Intensity of capital adjustment costs. Defaults to 0.5.
            k_min (float, optional): Lower bound for capital. Defaults to 0.8.
            k_max (float, optional): Upper bound for capital. Defaults to 1.2.
        """
       
        # Intertemporal elasticity of substitution
        self.ies_a = ies_a
        self.ies_b = ies_b
        self.gamma = np.zeros(self.num_countries)
        for i in range(0,self.num_countries):
            self.gamma[i] = ies_a+i*(ies_b-ies_a)/(self.num_countries-1)

        # Discount factor
        self.beta = beta
        # Capital share of income
        self.zeta = zeta
        # Depreciation rate
        self.delta = delta
        # Persistence of TFP shocks
        self.rho_Z = rho_Z
        # Standard deviation of TFP shocks
        self.sig_E = sig_E
        # Intensity of capital adjustment costs
        self.kappa = kappa

        # Lower bound for capital
        self.k_min = k_min
        # Upper bound for capital
        self.k_max = k_max
        
        # Aggregate productivity
        self.A_tfp = (1.0-self.beta*(1.0-self.delta))/(self.zeta*self.beta)
        # Welfare weight
        self.pareto = self.A_tfp**(1.0/self.gamma)

        # Lower bound for TFP
        self.a_min = -0.8*self.sig_E/(1.0-self.rho_Z)
        # Upper bound for TFP
        self.a_max = 0.8*self.sig_E/(1.0-self.rho_Z)
        
        # set flag
        self.flag_set_param = True 
        
    def set_integral_rule(self,quadrature_type:str='monomials_power'):
        """Set numerical itegration/quadrature rule. This is fixed to 'monomials_power' until further development.

        Args:
            quadrature_type (str, optional): Type of quadrature rule. Defaults to 'monomials_power'.
        """
        
        #assert quadrature_type=='GH-quadrature' or quadrature_type=='monomials_2d' or quadrature_type=='monomials_power'
        assert quadrature_type=='monomials_power'
        assert self.flag_set_param==True    
    
        # Type of quadrature values
        self.quadrature_type = quadrature_type
        
        # Number of integration nodes
        self.num_integral_nodes = 2*self.num_shocks**2 + 1

        # Deviations in one dimension (note that the origin is row zero)
        z0 = np.zeros((self.num_integral_nodes,self.num_shocks))
        for i1 in range(self.num_shocks):
            z0[i1*2+1,i1] =  1.0
            z0[i1*2+2,i1] = -1.0

        i0 = 0
        # Deviations in two dimensions
        for i1 in range(self.num_shocks):
            for i2 in range(i1+1,self.num_shocks):
                z0[2*self.num_shocks+1+i0*4,i1] =  1.0
                z0[2*self.num_shocks+2+i0*4,i1] =  1.0
                z0[2*self.num_shocks+3+i0*4,i1] = -1.0
                z0[2*self.num_shocks+4+i0*4,i1] = -1.0
                z0[2*self.num_shocks+1+i0*4,i2] =  1.0
                z0[2*self.num_shocks+2+i0*4,i2] = -1.0
                z0[2*self.num_shocks+3+i0*4,i2] =  1.0
                z0[2*self.num_shocks+4+i0*4,i2] = -1.0
                i0 += 1

        # Nodes
        integral_nodes                          = np.zeros((self.num_integral_nodes,self.num_shocks))
        integral_nodes[1:self.num_shocks*2+1,:] = z0[1:self.num_shocks*2+1,:]*np.sqrt(2.0+self.num_shocks)*self.sig_E
        integral_nodes[self.num_shocks*2+1:]    = z0[self.num_shocks*2+1:]*np.sqrt((2.0+self.num_shocks)/2.0)*self.sig_E

        # Weights
        integral_weights                        = np.zeros((self.num_integral_nodes))

        integral_weights[0]                     = 2.0/(2.0+self.num_shocks)
        integral_weights[1:self.num_shocks*2+1] = (4-self.num_shocks)/(2*(2+self.num_shocks)**2)
        integral_weights[self.num_shocks*2+1:]  = 1.0/(self.num_shocks+2)**2
        
        self.integral_nodes    = integral_nodes
        self.integral_weights  = integral_weights
        self.flag_set_integral = True 
                              
    def system_of_equations(self,x:np.array,state:np.array,grid:object)->np.array:
        """AI is creating summary for system_of_equations

        Args:
            x (np.array): [description]
            state (np.array): The values of the state variables
            grid (object):  The policiy function interpolant

        Returns:
            np.array: [description]
        """

        # State variables
        capStates = state[0:self.num_countries]
        tfpStates = state[self.num_countries:]

        # Policy values
        capPolicies = x[0:self.num_countries]
        lamb        = x[self.num_countries]

        if self.irbc_type == 'non-smooth':
            gz_alphas = x[self.num_countries+1:]

            # Garcia-Zengwill transformation of the occasionally binding constraints
            gz_alpha_plus  = np.maximum(0.0, gz_alphas)
            gz_alpha_minus = np.maximum(0.0,-gz_alphas)

        # Computation of integrands
        Integrands = self.expectation_of_FOC(capPolicies, state, grid)

        IntResult  = np.empty(self.num_countries)

        for i in range(self.num_countries):
            IntResult[i] = np.dot(self.integral_weights,Integrands[:,i])

        res = np.zeros(self.num_policies)

        # Computation of residuals of the equilibrium system of equations

        if self.irbc_type=='non-smooth':
            # Euler equations & GZ alphas
            for ires in range(0,self.num_countries):
                res[ires] = (self.beta*IntResult[ires] + gz_alpha_plus[ires])\
                                /(1.0 + self.AdjCost_ktom(capStates[ires],capPolicies[ires])) - lamb
                res[self.num_countries+1+ires] = capPolicies[ires] - capStates[ires]*(1.0-self.delta) - gz_alpha_minus[ires]
        else:
            # Euler equations
            for ires in range(0,self.num_countries):
                res[ires] = self.beta*IntResult[ires]/(1.0 + self.AdjCost_ktom(capStates[ires],capPolicies[ires])) - lamb


        # Aggregate resource constraint
        for ires2 in range(0,self.num_countries):
            res[self.num_countries] += self.F(capStates[ires2],tfpStates[ires2]) + (1.0-self.delta)*capStates[ires2] - capPolicies[ires2]\
                                - self.AdjCost(capStates[ires2],capPolicies[ires2])\
                                - (lamb/self.pareto[ires2])**(-1.0/self.gamma[ires2])
 

        return res    

    def expectation_of_FOC(self,ktemp:np.array, state:np.array, grid:object)->np.array:
        """Compute the expectation of the terms in the Euler equations of each country.

        Args:
            ktemp (np.array): The values for the capital policies
            state (np.array): The values of the state variables
            grid (object): Interpolant of the policy function

        Returns:
            np.array: The expectation terms for each country in each possible state tomorrow
        """

        # 1) Determine next period's tfp states

        new_state = np.zeros((self.num_integral_nodes,self.num_countries))

        for itfp in range(self.num_countries):
            new_state[:,itfp] = self.rho_Z*state[self.num_countries+itfp] + (self.integral_nodes[:,itfp] + self.integral_nodes[:,self.num_shocks-1])
            new_state[:,itfp] = np.where(new_state[:,itfp] > self.a_min, new_state[:,itfp], self.a_min)
            new_state[:,itfp] = np.where(new_state[:,itfp] < self.a_max, new_state[:,itfp], self.a_max)

        # 2) Determine next period's state variables
        evalPt                         = np.zeros((self.num_integral_nodes,self.num_countries*2))
        evalPt[:,0:self.num_countries] = ktemp
        evalPt[:,self.num_countries:]  = new_state

        # 3) Determine relevant variables within the expectations operator
        fval    = grid.eval(evalPt)
        capPrPr = fval[:,0:self.num_countries]
        lambPr  = fval[:,self.num_countries]
        #capPrPr = grid.evaluateBatch(evalPt)[:,0:self.num_countries]
        #lambPr  = grid.evaluateBatch(evalPt)[:,self.num_countries]

        if self.irbc_type=='non-smooth':
            #gzAlphaPr = grid.evaluateBatch(evalPt)[:,self.num_countries+1:]
            gzAlphaPr = fval[:,self.num_countries+1:]
            gzAplusPr = np.maximum(0.0,gzAlphaPr)

        # Compute tomorrow's marginal productivity of capital
        MPKtom = np.zeros((self.num_integral_nodes,self.num_countries))
        for impk in range(self.num_countries):
            MPKtom[:,impk] = 1.0 - self.delta + self.Fk(ktemp[impk],new_state[:,impk]) - self.AdjCost_k(ktemp[impk],capPrPr[:,impk])


        density = 1.0

        #Specify Integrand
        val = np.zeros((self.num_integral_nodes,self.num_countries))

        if self.irbc_type=='non-smooth':
            for iexp in range(self.num_countries):
                val[:,iexp] = (MPKtom[:,iexp]*lambPr - (1.0-self.delta)*gzAplusPr[:,iexp]) * density

        else:
            for iexp in range(self.num_countries):
                val[:,iexp] = MPKtom[:,iexp]*lambPr * density


        return val

    def print_parameters(self):
        """Print the parameters set by set_parameters.
        """

        H=['Parameter','Variable','Value']
        T=[]
        T.append(['Intertemporal elasticity of substitution(IES)','gamma','ies_a+(i-1)(ies_b-ies_a)/(N-1)'])
        T.append(['IES factor a','ies_a',self.ies_a])
        T.append(['IES factor b','ies_b',self.ies_b])
        T.append(['Discount factor','beta',self.beta])
        T.append(['Capital share of income','zeta',self.zeta])
        T.append(['Depreciation rate','delta',self.delta])         
        T.append(['Persistence of total factor productivity shocks','rho_Z',self.rho_Z])
        T.append(['Standard deviation of total factor productivity shocks','sig_E',self.sig_E])                     
        T.append(['Intensity of capital adjustment costs','kappa',self.kappa])     
        T.append(['Lower bound for capital','k_min',self.k_min])     
        T.append(['Upper bound for capital','k_max',self.k_max])     
        T.append(['Aggregate productivity','A_tfp',self.A_tfp])     
        T.append(['Welfare weight','pareto',self.pareto])    
        T.append(['Lower bound for total factor productivity','a_min',self.a_min])    
        T.append(['Upper bound for total factor productivity','a_max',self.a_max])   
        
        print(tabulate(T,headers=H))
                  
    def error_sim(self,policy_funcion:object,N:int)->np.array:
        """Compute the error measures along the simulation path of the given policy function.


        Args:
            policy_funcion (object): Policy function.
            N (int): Number of steps in the simulation.

        Returns:
            np.array: [description]
        """

        state_current                                              = np.zeros(shape=(1,self.grid_dim))
        state_current[0,0:self.num_countries]                      = (self.k_min + self.k_max)/2
        state_current[0,self.num_countries:2*self.num_countries+1] = (self.a_min + self.a_max)/2

        error = np.zeros(shape=(N,self.num_countries))
        
        for t in range(0,N):
            shock_local  = np.random.normal(0,1,(self.num_countries))
            shock_global = np.random.normal(0,1,(1))
            
            captial_current      = state_current[0,0:self.num_countries]
            productivity_current = state_current[0,self.num_countries:2*self.num_countries]
            policy_current       = policy_funcion.eval(state_current)
            
            capital_next         = policy_current[0,0:self.num_countries]
            lambda_next          = policy_current[0,self.num_countries:self.num_countries+1]
            productivity_next    = self.rho_Z*productivity_current+ self.sig_E*(shock_local+shock_global)
            mu_current           = np.zeros(self.num_countries)
            if self.irbc_type=='non-smooth':
                mu_current       = policy_current[0,self.num_countries+1:2*self.num_countries+1]
                
            state_next           = np.concatenate([capital_next,productivity_next])

            #Compute Density
            density  = 1.0
            
            error_ee = np.empty(self.num_countries)
            error_ic = np.empty(self.num_countries)

            for i in range(0,self.num_integral_nodes):

                # E[ln a_{i,t}] = E[\rho ln a_{i,t-1} + \sigma (e_{i,t} + e_{N,t})]
                # note e_{N,t} is the global shock fixed to the last country
                productivity_next_expectation = self.rho_Z*productivity_next + (self.integral_nodes[i,0:self.num_countries] + self.integral_nodes[i,-1])
                state_next_expectation        = np.concatenate([capital_next,productivity_next_expectation]).reshape((1,-1))

                policy_next_expectation  = policy_funcion.eval(state_next_expectation)
                capital_next_expectation = policy_next_expectation[0,0:self.num_countries]
                lambda_next_expectation  = policy_next_expectation[0,self.num_countries]
                
                marginal_cost_of_captial_next = 1.0 - self.delta + self.Fk(capital_next,productivity_next_expectation) - self.AdjCost_k(capital_next,capital_next_expectation)

                if self.irbc_type=='non-smooth':
                    mu_next_expectation  = policy_next_expectation[0,self.num_countries+1:2*self.num_countries+1]
                    temp = (lambda_next_expectation*marginal_cost_of_captial_next - (1.0-self.delta)*mu_next_expectation) * density
                else:
                    temp = lambda_next_expectation*marginal_cost_of_captial_next * density
    
                error_ee = error_ee + temp*self.integral_weights[i]
                
            error_ee = self.beta*error_ee/(lambda_next*(1.0+self.AdjCost_ktom(captial_current,capital_next))) - 1.0
            error_ic = 1.0 - captial_current/(capital_next*(1.0-self.delta))
            
            if self.irbc_type=='non-smooth':
                for i in range(0,self.num_countries):
                    error[t,i]= max(error_ee[i],error_ic[i],np.minimum(-error_ee[i],-error_ic[i]))
            else:
                error[t,:] = error_ee
                
            #update current state
            state_current = state_next.reshape((1,-1))
            
        return error


    def F(self,capital:np.array,sh:np.array)->np.array:
        """ Production function 

        Args:
            capital (np.array): Capital
            sh (np.array): Productivity shock

        Returns:
            np.array: Production
        """

        val = self.A_tfp * np.exp(sh)*np.maximum(capital,1e-6)**self.zeta

        return val

    def Fk(self,capital:np.array,sh:np.array)->np.array:
        """Marginal product of capital   

        Args:
            capital (np.array): Capital
            sh (np.array): Productivity shock

        Returns:
            np.array: Marginal product of capital   
        """
        val =  self.A_tfp * self.zeta*np.exp(sh)*np.maximum(capital,1e-6)**(self.zeta-1.0)

        return val
    
    def AdjCost(self,ktod:np.array,ktom:np.array)->np.array:
        """Capital adjustment cost  

        Args:
            ktod (np.array): Captial today 
            ktom (np.array): Captial tommorow 

        Returns:
            np.array: Capital adjustment cost 
        """

        captod = np.maximum(ktod,1e-6)
        captom = np.maximum(ktom,1e-6)

        j = captom/captod - 1.0
        val = 0.5 * self.kappa * j * j * captod

        return val

    def AdjCost_k(self,ktod:np.array,ktom:np.array)->np.array:
        """Derivative of capital adjustment cost w.r.t today's cap stock   

        Args:
            ktod (np.array): Captial today 
            ktom (np.array): Captial tommorow 

        Returns:
            np.array: Derivative of capital adjustment cost w.r.t today's cap stock 
        """

        captod = np.maximum(ktod,1e-6)
        captom = np.maximum(ktom,1e-6)

        j = captom/captod - 1.0
        j1 = captom/captod + 1.0
        val = (-0.5)*self.kappa*j*j1

        return val

    def AdjCost_ktom(self,ktod:np.array,ktom:np.array)->np.array:
        """Derivative of capital adjustment cost w.r.t tomorrows's cap stock 

        Args:
            ktod (np.array): Captial today 
            ktom (np.array): Captial tommorow 

        Returns:
            np.array: Derivative of capital adjustment cost w.r.t tomorrows's cap stock 
        """

        captod = np.maximum(ktod,1e-6)
        captom = np.maximum(ktom,1e-6)

        j = captom/captod - 1.0
        val = self.kappa * j


        return val

    def ARC_zero(self,lam_gues,gridPt)->float:
        """ Residual of aggregate resource constraint, used compute an initial guess for the ARC multiplier.

        Args:
            lam_gues ([type]): [description]
            gridPt ([type]): [description]

        Returns:
            float: [description]
        """
        
        res = 0.0
        
        for i1 in range(self.num_countries):
            res += np.exp(gridPt[self.num_countries+i1])*self.A_tfp*gridPt[i1]**self.zeta - (-self.delta*self.kappa/2.0)**2 - (lam_gues/self.pareto[i1])**(-self.gamma[i1])
        
        return res
        
    
