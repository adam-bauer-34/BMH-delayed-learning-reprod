"""Tree class.

Adam Michael Bauer
University of Illinois Urbana-Champaign
World Bank Group
10.2.2023
"""

import os

import numpy as np

from scipy.optimize import root
from scipy.integrate import quad 

class TreePathDep():
    """Tree class

    This class contains the attributes of the information tree used in the N-period optimization model.

    Attributes
    ----------
    N_periods: int
        number of periods
    
    base: int
        number of nodes to break into per learning point

    var_mean: float
        average value of variable distribution
    
    var_std: float
        standard deviation of variable distribution

    method: string
        options for filling out tree data; valid are:
            - "MC": distribution of data is sampled using Monte Carlo
            - "GHQ": distriution of data is filled in using Gauss-Hermite quadrature

    full_data: dict
        parameter values in each period at each node

    full_probs: dict
        parameter probabilities in each period at each node

    Properties
    ----------
    N_nodes_per_period: list
        number of nodes in each period
        generally: nodes_per_period = base^(period - 1)

    Methods
    -------
    initialize_tree_data:
        generates attriutes full_data and full_probs from variable distribution
    """

    def __init__(self, N_periods, base, var_mean, var_std, method='MC'):
        self.N_periods = N_periods
        self.base = base
        self.var_mean = var_mean
        self.var_std = var_std
        self.method = method

        if self.N_periods-1 > 1 and self.method == "GHQ":
            raise ValueError("Gauss-Hermite quadrature methods are only valid for one decision period, i.e., one learning point. Either use Monte Carlo sampling methods (set method='MC') or set the number of periods to 1.")

    @property
    def N_nodes_per_period(self):
        N_nodes_per_period = []
        for i in range(len(self.full_data.keys())):
            N_nodes_per_period.append(len(self.full_data[i]))
        return N_nodes_per_period

    def initialize_tree_data(self, check_generated_data=False, trunc_percentile=None):
        """Initialize tree data.

        Fill in the tree data.

        Parameters
        ----------
        check_generated_data: bool (default False)
            prints big message to report the generated data

        trunc_percentile: int (default None)
            percentile to truncate distribution at for GHQ_TRUNC method.
            raises error if not passed
        """
        if self.method == "MC":
            self._initialize_tree_data_MC(check_generated_data)
        
        elif self.method == "MC_TRUNC":
            if trunc_percentile == None:
                raise ValueError("No truncation percentile passed. Please add to initialize_tree_data function call if using MC_TRUNC method for tree data.")
            
            self._initialize_tree_data_MC_TRUNC(check_generated_data, trunc_percentile)


        elif self.method == "GHQ":
            self._initialize_tree_data_GHQ(check_generated_data)

        elif self.method == "GHQ_TRUNC":
            if trunc_percentile == None:
                raise ValueError("No truncation percentile passed. Please add to initialize_tree_data function call if using GHQ_TRUNC method for tree data.")
            
            self._initialize_tree_data_GHQ_TRUNC(check_generated_data, trunc_percentile)

        else:
            raise ValueError("Invalid setup method. Specify MC -- for Monte Carlo -- or GHQ -- for Gauss-Hermite quadrature.")

    def _initialize_tree_data_MC(self, check_generated_data=False):
        """Generate path independent tree with data from a distribution.

        This function creates two dictionaries, both of which store data and probabilities
        related to an assumed Gaussian distribution of a variable. The final period data
        is generated from the distribution, and the data at each prior period is found
        by taking expectations using backward induction.

        Parameters
        ----------
        check_generated_data: bool (False)
            print out generated data tree as a check
        """
        
        # make empty dictionaries to populate
        self.full_data = {}
        self.full_probs = {}

        N_periods_loop = self.N_periods - 1 # we only add nodes for periods not including the first

        # number of final states in the tree, dictates how many draws from the distribution we take
        N_final_states = self.base**N_periods_loop
        
        # make data for final period
        end_data = np.random.normal(self.var_mean, self.var_std, size=N_final_states)
        end_data = sorted(end_data)[::-1]
        end_probs = [1/len(end_data)] * len(end_data)

        # check generated data and probabilities if you would like
        if check_generated_data:
            print("Sorted end data: ", end_data)
            print("End probs: ", end_probs)

        # loop through periods backwards and populate the full_data and full_probs dictionaries
        for per in range(N_periods_loop, -1, -1):
            # if we're in the final period, set the data and probs equal to end period values
            if per == N_periods_loop:
                self.full_data[per] = end_data
                self.full_probs[per] = end_probs
            
            # if we're in the first period, we simply use the expectation of the entire distribution
            elif per == 0:
                self.full_data[per] = [self.var_mean]
                self.full_probs[per] = [1.0]
            
            # if we're in an intermediate period, use *next period*'s values to compute expectations of what
            # the value would be at that node
            else:
                # total number of nodes in next period
                next_period_total_states = self.base**(per + 1)
                
                # make temporary indices to fill
                tmp_ind_1 = 0
                tmp_ind_2 = self.base
                
                # make empty arrays to populate data
                tmp_data = []
                tmp_probs = []

                # loop through the upper bound index (tmp_ind_2) until we've exceeded the number of 
                # states in the next period. that's how we know we're done.
                while tmp_ind_2 < next_period_total_states+1:
                    # values and probabilities at range of next nodes to be used in the expectation
                    vals = self.full_data[per+1][tmp_ind_1:tmp_ind_2]
                    probs = self.full_probs[per+1][tmp_ind_1:tmp_ind_2]
                    
                    # take expectation
                    exp_val = sum([probs[i] * vals[i] for i in range(len(vals))])/ sum(probs)
                    tmp_data.append(exp_val)
                    tmp_probs.append(sum(probs))
                    
                    # index temporary indices
                    tmp_ind_1 += self.base
                    tmp_ind_2 += self.base
                
                # NORMALIZE PROBABILITIES TO ONE
                tmp_probs_norm = [prob/sum(tmp_probs) for prob in tmp_probs]
                
                # populate final dictionaries
                self.full_data[per] = tmp_data
                self.full_probs[per] = tmp_probs_norm

    def _initialize_tree_data_MC_TRUNC(self, check_generated_data=False, trunc_percentile=None):
        """Generate path independent tree with data from a distribution.

        This function creates two dictionaries, both of which store data and probabilities
        related to an assumed Gaussian distribution of a variable. The final period data
        is generated from the distribution, and the data at each prior period is found
        by taking expectations using backward induction.

        We use the Cai and Judd (2015) methodology, see:
        https://doi.org/10.1007/s00186-015-0495-z

        Parameters
        ----------
        check_generated_data: bool (False)
            print out generated data tree as a check
        """
        
        # make empty dictionaries to populate
        self.full_data = {}
        self.full_probs = {}

        N_periods_loop = self.N_periods - 1 # we only add nodes for periods not including the first

        # number of final states in the tree, dictates how many draws from the distribution we take
        N_final_states = self.base**N_periods_loop

        # find transformation to cutoff distribution
        # use sample average approximation to find the percentile value 
        # (i.e., the raw number, rather than the percentile number)
        perc_cutoff_samp = [np.percentile(np.random.normal(self.var_mean, self.var_std, int(1e5)), trunc_percentile) for i in range(1000)]
        
        # take avg
        perc_cutoff = np.mean(perc_cutoff_samp)

        # transformation factor for cutoff distribution
        self.trans_factor = (self.var_mean - perc_cutoff) * self.var_std**(-1)

        # find 'k' factor for the transformation
        self.k = self._get_scale(bound=self.trans_factor)
        
        # make data for final period
        x = np.random.normal(0, 1, size=N_final_states)
        end_data = self.var_mean + (1 - np.exp(-self.k * x)) * (1 + np.exp(-self.k * x))**(-1) * self.trans_factor * self.var_std
        end_data = sorted(end_data)[::-1]
        end_probs = [1/len(end_data)] * len(end_data)

        # check generated data and probabilities if you would like
        if check_generated_data:
            print("Sorted end data: ", end_data)
            print("End probs: ", end_probs)

        # loop through periods backwards and populate the full_data and full_probs dictionaries
        for per in range(N_periods_loop, -1, -1):
            # if we're in the final period, set the data and probs equal to end period values
            if per == N_periods_loop:
                self.full_data[per] = end_data
                self.full_probs[per] = end_probs
            
            # if we're in the first period, we simply use the expectation of the entire distribution
            elif per == 0:
                self.full_data[per] = [self.var_mean]
                self.full_probs[per] = [1.0]
            
            # if we're in an intermediate period, use *next period*'s values to compute expectations of what
            # the value would be at that node
            else:
                # total number of nodes in next period
                next_period_total_states = self.base**(per + 1)
                
                # make temporary indices to fill
                tmp_ind_1 = 0
                tmp_ind_2 = self.base
                
                # make empty arrays to populate data
                tmp_data = []
                tmp_probs = []

                # loop through the upper bound index (tmp_ind_2) until we've exceeded the number of 
                # states in the next period. that's how we know we're done.
                while tmp_ind_2 < next_period_total_states+1:
                    # values and probabilities at range of next nodes to be used in the expectation
                    vals = self.full_data[per+1][tmp_ind_1:tmp_ind_2]
                    probs = self.full_probs[per+1][tmp_ind_1:tmp_ind_2]
                    
                    # take expectation
                    exp_val = sum([probs[i] * vals[i] for i in range(len(vals))])/ sum(probs)
                    tmp_data.append(exp_val)
                    tmp_probs.append(sum(probs))
                    
                    # index temporary indices
                    tmp_ind_1 += self.base
                    tmp_ind_2 += self.base
                
                # NORMALIZE PROBABILITIES TO ONE
                tmp_probs_norm = [prob/sum(tmp_probs) for prob in tmp_probs]
                
                # populate final dictionaries
                self.full_data[per] = tmp_data
                self.full_probs[per] = tmp_probs_norm

    def _initialize_tree_data_GHQ(self, check_generated_data=False):
        """Generate path independent tree with data from a distribution.

        This function creates two dictionaries, both of which store data and probabilities
        related to an assumed Gaussian distribution of a variable. This method uses an unbounded
        Gauss-Hermite quadrature method to approximate the tree data.

        Parameters
        ----------
        check_generated_data: bool (False)
            print out generated data tree as a check
        """
        
        # make empty dictionaries to populate
        self.full_data = {}
        self.full_probs = {}

        # find roots of Hermite polynomials to evaluate distribution at
        roots = self._gen_ghq_roots()

        # compute weights via the usual formula
        self.weights = self._gen_ghq_weights(roots)

        # translate roots to relevant interval
        self.trans_roots = self._get_trans_roots(roots)

        # check generated data and probabilities if you would like
        if check_generated_data:
            print("Sorted data: ", self.trans_roots)
            print("Weights: ", self.weights)
        
        # populate dictionaries
        self.full_data[0] = np.array([self.var_mean])
        self.full_probs[0] = np.array([1.0])

        self.full_data[1] = self.trans_roots
        self.full_probs[1] = self.weights

    def _gen_ghq_roots(self):
        """Import Gauss-Hermite quadrature roots.
        """

        cwd = os.getcwd()
        ghq_file = cwd + "/data/cal/ghq_roots.csv"
        roots = np.genfromtxt(ghq_file, delimiter=',')[self.base-1]
        return roots[~np.isnan(roots)]
    
    def _gen_ghq_weights(self, roots):
        """Using Hermite polynomials, generate weights.
        """
        hermite_vals = self._get_hermite(roots, self.base-1)
        w_i = 2**(self.base-1) * np.math.factorial(self.base) * np.sqrt(np.pi) * (self.base**2 * hermite_vals**2)**(-1)
        return w_i * np.sqrt(np.pi)**(-1)

    def _get_hermite(self, x, N):
        """Get Hermite polynomial basis functions at point.
        NOTE: we use physics Hermite polynomials!

        Parameters
        ----------
        x: list
            grid point to evaluate H_n(x) at

        n: int
            order of Hermite polynomial to evaluate

        Returns
        -------
        vals: list
            H_n(x) values
        """

        vals = np.zeros(len(x))
        for i in range(len(x)):
            if N == 0:
                vals[i] = 1.0
            elif N == 1:
                vals[i] = 2 * x[i]
            else:
                # store previous two Hermite polynomial values
                herm_vals = [1.0, 2*x[i]]

                # use recursion relation to find remaining values
                ni = 2
                while ni <= N:
                    tmp = 2 * (x[i] * herm_vals[-1] - (ni - 1) * herm_vals[-2])
                    herm_vals.append(tmp)
                    ni += 1
                vals[i] = herm_vals[-1]

        return vals

    def _get_trans_roots(self, roots):
        return self.var_mean + np.sqrt(2) * self.var_std * roots

    def _initialize_tree_data_GHQ_TRUNC(self, check_generated_data=False, trunc_percentile=None):
        """Generate path independent tree with data from a distribution.

        This function creates two dictionaries, both of which store data and probabilities
        related to an assumed Gaussian distribution of a variable. This method uses an unbounded
        Gauss-Hermite quadrature method to approximate the tree data.

        We use the Cai and Judd (2015) methodology, see:
        https://doi.org/10.1007/s00186-015-0495-z

        Parameters
        ----------
        check_generated_data: bool (False)
            print out generated data tree as a check
        
        trunc_percentile: int
            percentile to cutoff distribution at
        """
        
        # make empty dictionaries to populate
        self.full_data = {}
        self.full_probs = {}

        # find roots of Hermite polynomials to evaluate distribution at
        roots = self._gen_ghq_roots()

        # compute weights via the usual formula
        self.weights = self._gen_ghq_weights(roots)

        # find transformation to cutoff distribution
        # use sample average approximation to find the percentile value 
        # (i.e., the raw number, rather than the percentile number)
        perc_cutoff_samp = [np.percentile(np.random.normal(self.var_mean, self.var_std, int(1e5)), trunc_percentile) for i in range(1000)]
        
        # take avg
        perc_cutoff = np.mean(perc_cutoff_samp)

        # transformation factor for cutoff distribution
        self.trans_factor = (self.var_mean - perc_cutoff) * self.var_std**(-1)

        # find 'k' factor for the transformation
        self.k = self._get_scale(bound=self.trans_factor)

        # transform roots following the Cai and Judd transformation
        self.trans_roots_cutoff = self._get_cutoff_roots(roots, self.trans_factor, self.k)

        # check generated data and probabilities if you would like
        if check_generated_data:
            print("Sorted data: ", self.trans_roots_cutoff)
            print("Weights: ", self.weights)
        
        self.full_data[0] = np.array([self.var_mean])
        self.full_probs[0] = np.array([1.0])

        self.full_data[1] = self.trans_roots_cutoff
        self.full_probs[1] = self.weights
    
    def _get_scale(self, bound):
        sol = root(self._diff, x0=[0.5], args=[bound])
        return sol.x[0]

    def _diff(self, scale, args):
        bound, = args
        piece = bound**2 * quad(self._integrand, -30, 30, args=[scale])[0]
        return piece - 1

    def _integrand(self, x, args):
        scale, = args
        y = 2 * (1 + np.exp(-scale * x))**(-1) - 1
        z = y * y * np.exp(-1 * x * x * 0.5) * np.sqrt(2 * np.pi)**(-1)
        return z

    def _get_cutoff_roots(self, x, scaling_factor, k):
        x = np.array(x)
        return self.var_mean + scaling_factor * (1 - np.exp(-k * x)) * (1 + np.exp(-k * x))**(-1) * self.var_std

"""
t = TreePathDep(1, 8, 770, 220, 'GHQ_TRUNC')

t.initialize_tree_data(trunc_percentile=1)

print(t.trans_factor)
"""