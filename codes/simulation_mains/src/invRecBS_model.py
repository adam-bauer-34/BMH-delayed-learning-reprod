"""Abatement Investment model class with recourse.

Adam Michael Bauer
University of Illinois Urbana-Champaign
World Bank Group
9.14.2023
"""

import os 

import numpy as np 
import cvxpy as cp 
import xarray as xr
import pandas as pd

from datatree import DataTree
from .tree import TreePathDep
from .invBase_model import INVBaseModel

class INVRecourseModelWithBS():
    """Abatement investment model class with recourse and backstop technology.

    This class contains the necessary attributes and methods for solving the
    abatement investment model with recourse (INVRec model) using `cvxpy`.

    Attributes
    ----------
    cal: string
        name of calibration 

    sectors: list
        list of sectors by name (list of strings)
    
    N_secs: int
        number of sectors
    
    r: `cvxpy.Parameter`, float
        social discount rate
    
    beta: `cvxpy.Parameter`, float
        discount factor, \beta := (1+r)^-1
    
    B: `cvxpy.Parameter`, float
        remaining carbon budget
    
    T: `cvxpy.Parameter`, float
        terminal time horizon

    dt: `cvxpy.Parameter`, float
        time discretization

    times: (T/dt,) numpy array
        list of times the model is solved at
    
    betas: (len(times),) numpy array
        \beta^t for all t \in times

    abars: `cvxpy.Parameter`, (N_secs,) array
        emissions rates in each sector

    gbars: `cvxpy.Parameter`, (N_secs,) list
        marginal abatement cost coefficient in each sector

    cbars: `cvxpy.Parameter`, (N_secs,) list (default=None)
        investment cost coefficient in each sector

    deltas: `cvxpy.Parameter`, (N_secs,) list
        capital depreciation rate in each sector

    psi_0: `cvxpy.Parameter`, float
        initial amount of cumulative emissions

    a_0s: `cvxpy.Parameter`, (N_secs,) list
        initial amount of abatmenet in each sector

    x: `cvxpy.Variable`, (N_secs, len(times))
        abatement investment in each sector
    
    a: `cvxpy.Variable`, (N_secs, len(times)+1)
        abatement rate
    
    psi: `cvxpy.Variable`, (len(times)+1)
        cumulative emissions

    F_psi: `cvxpy.Variable`, (len(times),)
        flux in psi equation of motion, required to extract the
        SCC

    psi_vec: `cvxpy.Variable`, (len(times),)
        cumulative emissions beyond t=0
    
    flux_rhs_v: `cvxpy.Variable`, (len(times),)
        right hand side of flux equation to set F_psi equal to

    constraints: list
        set of constraint expressions for optimization

    prob: `cvxpy.Problem`
        optimization problem

    Methods
    -------
    _import_calibration:
        imports  calibration based off of self.cal parameter

    _cal_model:
        calibrates model using root finding algorithm

    _get_diff:
        finds difference between INV model run and equivalent MAC model run
        
    solve_opt:
        solves optimization problem with specified parameters
    
    _save_output:
        saves output of optimization to netCDF

    _get_flux_rhs:
        get right hand side of flux for cumulative emissions equation of motion

    _get_psi_vector:
        generate the vector of cumulative emissions for t>0

    _get_abatement_vector:
        generate a vector of abatement values for t>0

    _get_delta_matrix:
        generate capital depreciations for abatement investment for t < T
    """

    def __init__(self, cal, rec_cal, scale=1.):
        self.cal = cal # calibration name
        self.rec_cal = rec_cal
        self.scale = float(scale)

        # if the calibration name is not a string, throw an error
        if not isinstance(self.cal, str) and not ininstance(self.rec_cal, str):
            raise ValueError("Calibration name must be a string. You passed {} and {}, which is type {} and {}.".format(self.cal, self.rec_cal, type(self.cal), type(self.rec_cal)))

        self._import_calibration()

        # big ol print message
        print("\n\nModel import successful. This is an abatement investment model with recourse and backstop technologies.")
        print("\n-------------------\nGLOBAL PARAMETERS\n-------------------\n")
        print("Calibration name: {}.".format(self.cal))
        print("Time horizon: {} yrs.".format(self.T.value))
        print("Social discount rate: {}%/yr.".format(self.r.value*100))
        print("Expected remaining carbon budget: {} GtCO2.".format(self.B.value))
        print("Remaining carbon budget standard deviation: {} GtCO2.".format(self.B_std))
        print("Time discretization (dt): {} yrs.".format(self.dt.value))
        print("\n-------------------\nSECTORAL PARAMETERS\n-------------------\n")
        print(''.join(["SECTOR: {}. EMISSIONS RATE: {} GtCO2/yr. DEPRECIATION RATE: {}%/yr.  COST COEFFICIENT: {} ($/tCO2) / (GtCO2/yr^3).\n".format(i,j,k,l) for i,j,k,l in zip(self.sectors, self.abars.value, self.deltas.value*100, self.cbars.value)]))
        print("\n-------------------\nRECOURSE PARAMETERS\n-------------------\n")
        print("Recourse calibration name: {}.".format(self.rec_cal))
        print("Information revelation times (after the t=0): {} yrs.".format(self.rev_times))
        print("This is a {}-period optimization problem embedded in a base-{} path dependent tree structure.\n".format(self.N_periods, self.base))

    def _import_calibration(self):
        """Import calibration.

        See the TAXONOMY. The MACRecourseModel class is passed a string for which calibration
        of the model we're using.
        """

        # get current working directory and make two paths to data
        cwd = os.getcwd()
        path_glob = ''.join([cwd, '/data/cal/', self.cal, '_glob.csv'])
        path_secs = ''.join([cwd, '/data/cal/', self.cal, '_secs.csv'])
        path_rec = ''.join([cwd, '/data/cal/rec_', self.rec_cal, '.csv'])

        # import csvs using Pandas
        df_glob = pd.read_csv(path_glob, ',', header=0)
        df_secs = pd.read_csv(path_secs, ',', header=0, index_col=0)
        df_rec = pd.read_csv(path_rec, ',', header=0)

        # parse global parameters
        self.r = cp.Parameter(nonneg=True, 
                              value=df_glob['r'].values[0]) # social discount rate
        self.beta = (1+self.r)**(-1) # discount factor 
        self.B = cp.Parameter(nonneg=True,
                              value=df_glob['B'].values[0]) # remaining carbon budget
        self.B_std = df_glob['B_std'].values[0] # remaining carbon budget standard deviation
        self.T = cp.Parameter(nonneg=True, 
                              value=df_glob['T'].values[0]) # terminal time horizon
        self.dt = cp.Parameter(nonneg=True,
                               value=df_glob['dt'].values[0]) # time discretization
        self.psi_0 = cp.Parameter(nonneg=True, 
                                  value=df_glob['psi_0'].values[0]) # initial conditions for cumulative emissions
        
        # parse sector-specific parameters
        self.sectors = df_secs.columns.values # sectors

        if not self.sectors.all() == 'DAC':
            raise Exception("No entries for direct air capture technologies (named 'DAC' or 'DirectAirCapture') detected in calibration file: {}_secs.csv.\n(You may want to check the name of the backstop technologies 'sector'; it must be 'DAC' or 'DirectAirCapture'.)".format(self.cal))

        self.N_secs = len(self.sectors) # number of sectors 
        self.abars = cp.Parameter(self.N_secs, nonneg=True, 
                                  value=df_secs.loc['abar'].values) # emissions rates by sector
        self.deltas = cp.Parameter(self.N_secs, nonneg=True, 
                                   value=df_secs.loc['delta'].values) # capital depreciations rate in each sector
        self.a_0s = cp.Parameter(self.N_secs, nonneg=True, 
                                 value=df_secs.loc['a_0'].values) # initial amount of abatement in each sector
        # NOTE: gbars is not a `cvxpy` parameter since we don't use it in the optimization in INV model
        self.gbars = df_secs.loc['gbar'].values # marginal abatement cost

        # try to import cbar parameters. if none exist, make them
        # NOTE: the KeyError comes from the fact that df_secs.loc['cbar'] has no entries
        try:
            self.cbars = cp.Parameter(self.N_secs, nonneg=True, 
                                      value=df_secs.loc['cbar'].values/self.scale)

        except KeyError:
            # make relative costs for calibration
            self._cal_model()

        # parse recourse parameters
        self.rev_times = df_rec['dec_times'].values
        
        if self.rev_times.any() > self.T.value or self.rev_times.any() < 0:
            raise Exception("Invalid learning times: {}.".format(self.rev_times))

        self.N_periods = len(self.rev_times) + 1 # number of model periods
        self.base = int(df_rec['base'].values[0])
        self.period_times = [0.0, *self.rev_times, self.T.value] # times for periods

                # get, and set, type of intergration method for the expectation operator in the objective function
        self.method = int(df_rec['method'].values[0]) # type of integration in expectation operator

        if self.method == 0:
            self.method = 'MC'
            self.trunc_percentile = None

        elif self.method == 1:
            self.method = "GHQ"
            self.trunc_percentile = None

        elif self.method == 2:
            self.method = "MC_TRUNC"
            self.trunc_percentile = df_rec['trunc_percentile'].values[0]

        elif self.method == 3:
            self.method = "GHQ_TRUNC"
            self.trunc_percentile = df_rec['trunc_percentile'].values[0]

        else:
            raise ValueError("Invalid method passed from recourse parameter file. Currently supported methods are:\n0 = 'MC' (Monte Carlo sampling)\n1 = 'GHQ' (Gauss-Hermite quadrature)\n2 = 'MC_TRUNC' (truncated Monte Carlo)\n3 = 'GHQ_TRUNC' (Gauss-Hermite quadrature, truncated)")

    def _cal_model(self):
        """Calibrate abatement investment model.

        The condition is that the total costs for the MAC model with
        equivalent conditions have the same costs.

        Parameters
        ----------
        save_cal: bool
            save calibration to csv?
        """

        # instantiate the base investment model **without backstop technologies**, 
        #which automatically calibrates if one does not exist and saves it with B = E(B)
        i = INVBaseModel(self.cal.replace('bs', ''))

        # cbar for DAC is the (gamma_DAC / gamma_Waste) * cbar_Waste
        cbars_val = np.hstack([i.cbars.value, self.gbars[-1] * self.gbars[0]**(-1) * i.cbars.value[0]])
        
        # set as cbars
        self.cbars = cp.Parameter(self.N_secs, nonneg=True, value=cbars_val)

        # save calibration to a csv
        df = pd.DataFrame([self.gbars, self.abars.value, self.deltas.value, self.a_0s.value, self.scale*self.cbars.value],
                              index=['gbar', 'abar', 'delta', 'a_0', 'cbar'], 
                              columns=self.sectors)

        cwd = os.getcwd()
        path = ''.join([cwd, '/data/cal/', self.cal, '_secs.csv'])
        df.to_csv(path)
        
        # re-run calibration function, where a calibration should now exist
        self._import_calibration()

    def tree_init(self, print_outcome=False):
        """Initialize information tree based on RCB parameters above.

        Parameters
        ----------
        print_outcome: bool (default: False)
            print the data once its generated?
        """

        print("Initializing data tree...")
        self.tree = TreePathDep(self.N_periods, self.base, self.B.value, self.B_std, method=self.method)
        self.tree.initialize_tree_data(trunc_percentile=self.trunc_percentile)
        print("Done!")

        if print_outcome:
            print("The tree data is: ")
            for per in self.tree.full_data.keys():
                print("\n-------------------\nPERIOD {}\n-------------------\n".format(per))
                print("Values: {}.\n".format(self.tree.full_data[per]))
                print("Number of nodes: {}.\n".format(self.tree.N_nodes_per_period[per]))
                print("Probabilities: {}.".format(self.tree.full_probs[per]))
            print("\n")

    def prob_init(self, print_outcome=False):
        """Initialize problem attributes.

        This function creates the cvxpy Problem class, the constraints, the objective function,
        and so on. We loop through each tree period and state to define constraints and objective terms.

        Parameters
        ----------
        print_outcome: bool (default = False)
            print the result of this process? 
            NOTE: this might not be particularly readable for humans, as cvxpy objects are a bit
            opaque. but one can, for example, check shapes of constraints and so on
        """

        self.times = {} # times
        self.betas = {} # discount factor
        self.x = {} # investment 
        self.a = {} # abatement
        self.psi = {} # cumulative emissions
        self.F_psi = {} # emissions flux

        # populate above dictionaries 
        for per in range(self.N_periods):
            self.times[per] = np.arange(self.period_times[per], self.period_times[per+1], self.dt.value)
            self.betas[per] = self.beta.value**self.times[per]
            self.x[per] = [cp.Variable((self.N_secs, len(self.times[per])), nonneg=True) for state in range(self.tree.N_nodes_per_period[per])]
            self.a[per] = [cp.Variable((self.N_secs, len(self.times[per]) + 1), nonneg=True) for state in range(self.tree.N_nodes_per_period[per])]
            self.psi[per] = [cp.Variable(len(self.times[per]) + 1, nonneg=True) for state in range(self.tree.N_nodes_per_period[per])]
            self.F_psi[per] = [cp.Variable(len(self.times[per]), nonneg=True) for state in range(self.tree.N_nodes_per_period[per])]

        # make objective function
        self.obj = 0.0
        for per in range(self.N_periods):
            tmp_obj = cp.sum([self.tree.full_probs[per][state] * self.betas[per]\
                              @ cp.sum([self.cbars[i].value * (2)**(-1) * self.x[per][state][i]**(2) for i in range(self.N_secs)])\
                              for state in range(self.tree.N_nodes_per_period[per])])
            self.obj += tmp_obj

        # generate constraints for each period and state therein
        self.constraints = []

        ## initial conditions of state variables
        self.constraints.append(self.psi[0][0][0] == self.psi_0)
        self.constraints.append(self.a[0][0][:, 0] == self.a_0s)

        ## on continuity between learning points in cumulative emissions and abatement 
        tmp_ind_1 = 0
        tmp_ind_2 = self.tree.base
        for per in range(self.N_periods - 1):
            for state in range(self.tree.N_nodes_per_period[per]):
                self.constraints.extend([self.psi[per][state][-1] == self.psi[per+1][tmp_ind_1:tmp_ind_2][i][0] for i in range(len(self.psi[per+1][tmp_ind_1:tmp_ind_2]))])
                for k in range(self.N_secs):
                    self.constraints.extend([self.a[per][state][k, -1] == self.a[per+1][tmp_ind_1:tmp_ind_2][i][k, 0] for i in range(len(self.psi[per+1][tmp_ind_1:tmp_ind_2]))])
                tmp_ind_1 += self.tree.base
                tmp_ind_2 += self.tree.base
            tmp_ind_1 = 0
            tmp_ind_2 = self.tree.base

        for per in range(self.N_periods):
            for state in range(self.tree.N_nodes_per_period[per]):
                ## cap on abatement
                self.constraints.extend([self.a[per][state][i, :] <= self.abars[i] for i in range(self.N_secs)])

                ## irreversibility of capital stocks 
                self.constraints.extend([self.x[per][state][i, :] - self.deltas[i] * self.a[per][state][i, :-1] >= 0 for i in range(self.N_secs)])
                
                if per == self.N_periods - 1:
                    ## cap the cumulative emissions in each period
                    self.constraints.append(self.psi[per][state][-1] <= self.tree.full_data[per][state])

                ## set non-initial cumulative emissions in each period
                tmp_psi_vec = self._get_psi_vector(per, self.times[per], 
                                                   self.psi[per][state],
                                                   self.F_psi[per][state])
                self.constraints.append(self.psi[per][state][1:] == tmp_psi_vec)

                ## set non-initial abatement in each period
                if per != 0:
                    self.constraints.extend([self._get_abatement_vector(per, k, self.times[per], self.times[per-1][-1],
                                         self.a[per][state][k, :], self.deltas[k].value, self.x[per][state][k, :]) == self.a[per][state][k, 1:] for k in range(self.N_secs)])
                
                else:
                    self.constraints.extend([self._get_abatement_vector(per, k, self.times[per], 0.0,
                                         self.a[per][state][k, :], self.deltas[k].value, self.x[per][state][k, :]) == self.a[per][state][k, 1:] for k in range(self.N_secs)])
                                        

        ## on the emissions flux (done last to make the scc more easily accessible later)
        for per in range(self.N_periods):
            for state in range(self.tree.N_nodes_per_period[per]):
                tmp_flux_vec = self._get_flux_rhs(self.times[per], self.a[per][state])
                self.constraints.append(self.F_psi[per][state] == tmp_flux_vec)

        # define problem
        self.prob = cp.Problem(cp.Minimize(self.obj), self.constraints)

        # print results if desired
        if print_outcome:
            print("Problem initialization successful. The outcome is: ")
            for per in self.tree.full_data.keys():
                print("\n-------------------\nPERIOD {}\n-------------------\n".format(per))
                print("Times: {}.\n".format(self.times[per]))
                print("Abatement: {}.\n".format(self.a[per]))
                print("Cumulative Emissions: {}.".format(self.psi[per]))
            print("\n")

    def _get_psi_vector(self, period, times, psi, psi_flux):
        """Get the vector of cumulative emissions for t > 0.

        This is essentially the cumulative emissions equation of motion evaluated
        at each point in time.

        Parameters
        ----------
        period: int
            what period we're in
        
        times: list
            list of times the period persists for

        psi: cp.Variable with length len(times) + 1
            cumulative emissions in the period & state considered

        psi_flux: cp.Variable with length len(times)
            emissions flux in the period & state considered
        
        Returns
        -------
        psi_vec: cp.Variable with length len(times)
            the full vector for the RHS of the cumulative emissions EoM
        """

        # first, the piece associated with initial conditions
        if period == 0:
            psi_init_piece = np.ones(len(times)) * self.psi_0.value
        else:
            psi_init_piece = np.ones(len(times)) * psi[0]

        # now the piece associated with the flux
        flux_piece = np.tri(len(times)) @ psi_flux

        # return equation of motion
        return psi_init_piece + self.dt.value * flux_piece
    
    def _get_flux_rhs(self, times, a):
        """Get the emissions flux. 

        Necessary to get SCC from optimizer. 
        flux = \sum_{sectors}{emis - abatement} for all t

        Parameters
        ----------
        times: list
            list of times the period persists for

        a: cp.Variable with shape (N_secs, len(times))
            the abatement in this period

        Returns
        -------
        flux: cp.Variable with shape len(times)
            the emissions flux
        """

        # emissions flux
        # sum_i (abar - a) - DAC_abatement
        flux = cp.sum(self.abars[:-1]) * np.ones(len(times)) - cp.sum(a[:-1, :-1], axis=0) - a[-1, :-1]
        return flux

    def _get_abatement_vector(self, period, sec, times, t_f_last_per, a_it, delta, x_it):
        """Get abatement vector for a given sector.

        Parameters
        ----------
        period: int
            current period

        sec: int
            current sector
            
        times: list
            list of times in this period

        a_it: `cvxpy.Variable` with length (len(times)+1,)
            abatement in the current sector in the current period

        delta: float
            capital depreciation rate in sector

        x_it: `cvxpy.Variable`, (len(times),) list
            investment in current sector

        Returns
        -------
        a_it: `cvxpy.Expression`
            the right hand side of the abatement vector's equation
        """

        # piece associated with initial conditions
        if period == 0:
            init_piece = (1.0 - delta * self.dt.value)**times * self.a_0s[sec]
        else:
            init_piece = (1.0 - delta * self.dt.value)**(times - t_f_last_per) * a_it[0]

        # piece associated with investment (i.e., the abatement flux)
        delta_matrix = self._get_delta_matrix(delta, times)
        flux_piece = delta_matrix @ x_it

        # sum & return
        a_it = init_piece + self.dt.value * flux_piece
        return a_it

    def _get_delta_matrix(self, delta, times):
        """Generate matrix of capital deprecation values for abatement
        investment.

        Parameters
        ----------
        delta: float
            abatement capital depreciation rate
        
        times: list
            list of times in this period

        Returns
        -------
        delta_matrix: (len(times), len(times)) matrix
            matrix of capital depreciation rates
        """

        delta_matrix = np.zeros((len(times), len(times)))
        for i in range(len(times)):
            for j in range(len(times)):
                if i == j:
                    delta_matrix[i,j] = 1.0
                elif j < i:
                    delta_matrix[i,j] = (1 - delta * self.dt.value)**(i-j)
                else:
                    continue
        return delta_matrix

    def solve_opt(self, save_output=True, verbose=True, 
                  cal_purposes=False, print_outcome=False, suppress_prints=False):
        """Solve optimization problem with `cvxpy`.

        Parameters
        ----------
        save_output: bool (default=True)
            save output of optimization to netCDF?

        verbose: bool (default=True)
            print verbose output from optimization?

        cal: bool (default=False)
            are we solving this for calibrating the INV model?
            (just suppresses the print statements at the end)

        print_outcome: bool (default=False)
            print DataTree object? **only happens if we save output**
        
        suppress_prints: bool (default=False)
            suppress all print statements from optimization? (cleans up terminal during risk premium calcs)
        """

        # solve problem
        if suppress_prints:
            self.prob.solve(solver=cp.GUROBI, verbose=False)
        
        else:
            self.prob.solve(solver=cp.GUROBI, verbose=verbose, OptimalityTol=1e-9, BarQCPConvTol=1e-12, BarConvTol=1e-12)

        # get statistics from solver 
        self.solve_stats = self.prob.solver_stats

        self._process_opt_output()

        # if we save the output, save it (so long as the optimizer returned the optimal result)
        if save_output and self.prob.status == 'optimal':
            self._save_output(print_outcome)
        
        elif cal_purposes and self.prob.status=='optimal':
            print("MAC model finished with optimal status.")

        elif suppress_prints and self.prob.status=='optimal':
            print("INV model finished with optimal status.")

        # else, print the outcome of the optimization
        else:
            print("\n---------------\nGLOBAL OUTCOME\n---------------\n")
            print("Status: ", self.prob.status)
            print("Total cost: ", self.prob.value)
            print("\n---------------\nRESULTS FOR DECISION AND STATE VARIABLES\n---------------\n")
            for sec in range(self.N_secs):
                print("\n---------------\nSECTOR: {}.\n---------------\n".format(self.sectors[sec]))
                for per in range(self.N_periods):
                    for state in range(self.tree.N_nodes_per_period[per]):
                        print("Period {} investment path(s): ".format(per), self.x_proc[per][state][sec])
                        print("Period {} abatement path(s): ".format(per), self.a_proc[per][state][sec], '\n')
            print("\n---------------\nAGGREGATE QUANTITIES (NON-SECTOR SPECIFIC)\n---------------\n")
            for per in range(self.N_periods):
                for state in range(self.tree.N_nodes_per_period[per]):
                    print("Period {} cumulative emissions path(s): ".format(per), self.psi_proc[per][state])

                    # the SCC is the final condition value
                    print("Period {} carbon price: ".format(per), self.scc_proc[per][state], '\n')
                
    def _process_opt_output(self, print_outcome=False):
        """Process model output.

        Loop through three and store model output as numpy arrays and dictionaries.
        """

        # make empty dictionaries
        self.x_proc = {}
        self.a_proc = {}
        self.psi_proc = {}
        self.scc_proc = {}

        # loop through periods and extract abatement and cumulative emissions
        for per in range(self.N_periods):
            self.x_proc[per] = [self.x[per][state].value for state in range(self.tree.N_nodes_per_period[per])]
            self.a_proc[per] = [self.a[per][state].value for state in range(self.tree.N_nodes_per_period[per])]
            self.psi_proc[per] = [self.psi[per][state].value for state in range(self.tree.N_nodes_per_period[per])]

        # loop through constraints and extract carbon price
        # NOTE: the last N constraints are the carbon price constraints, ordered by period
        # and then state. Hence we start at the furthest back constraint to get period = 0's
        # carbon price, and then go forward to get later period carbon prices.
        tmp_const_ind = -np.sum(self.tree.N_nodes_per_period)
        for per in range(self.N_periods):
            scc_proc_tmp = []
            for state in range(self.tree.N_nodes_per_period[per]):
                scc_proc_tmp.append(-1 * self.constraints[tmp_const_ind].dual_value * self.scale)
                tmp_const_ind += 1
            self.scc_proc[per] = scc_proc_tmp

        # xarray only likes lists, not dictionaries, which makes me sad 
        datatree_dict = {}

        for per in range(self.N_periods):
            tmp_states = [el for el in range(self.tree.N_nodes_per_period[per])]
            tmp_ds = xr.Dataset(data_vars={'investment': (['state', 'sector', 'time'], self.x_proc[per]),
                        'abatement': (['state', 'sector', 'time_state'], self.a_proc[per]),
                        'cumulative_emissions': (['state', 'time_state'], self.psi_proc[per]),
                        'scc': (['state', 'time'], self.scc_proc[per]),
                        'cbars': (['sector'], self.cbars.value * self.scale),
                        'abars': (['sector'], self.abars.value),
                        'deltas': (['sector'], self.deltas.value),
                        'a_0s': (['sector'], self.a_0s.value),
                        'B_tree': (['state'], self.tree.full_data[per]),
                        'B_probs': (['state'], self.tree.full_probs[per])},
                coords={'state': (['state'], tmp_states),
                        'time': (['time'], self.times[per]),
                        'time_state': (['time_state'], np.hstack([self.times[per], 1e10])),
                        'sector': (['sector'], self.sectors)})
            datatree_dict[str(per)] = tmp_ds

        # make DataTree object with parent coordinate named 'period'
        self.data_tree = DataTree.from_dict(datatree_dict, 'period')

        self.data_tree['0'].attrs = {'total_cost': self.prob.value * self.scale,
                         'r': self.r.value,
                         'beta':self.beta.value,
                         'dt': self.dt.value,
                         'T': self.T.value,
                         'B': self.B.value,
                         'B_std': self.B_std,
                         'psi_0': self.psi_0.value,
                         'status': self.prob.status,
                         'solve_time': self.solve_stats.solve_time,
                         'method': self.method,
                         'notes': "These are global simulation attributes, but owing to a limitation of the `datatree` package, they're stored under the first period."}
        
        if print_outcome:
            print(self.data_tree)

    def _save_output(self, print_outcome=False):
        """Save output of optimization to netCDF file.

        Parameters
        ----------
        print_outcome: bool (default=False)
            print datatree at the end?
        """

        # get current working directory, save output to data/output/ folder
        cwd = os.getcwd()
        path = ''.join([cwd, '/data/output/', self.cal, '_', self.method, '_invrec_output.nc'])
        self.data_tree.to_netcdf(filepath=path, mode='w', format='NETCDF4', engine='netcdf4')
        print("\nData successfully saved to file:\n{}".format(path))

    def calc_risk_premium(self):
        """Compute the risk premium of the recourse policy over the no uncertainty model.

        Risk premium is defined as: (t=0 cost of recourse model policy) - (t=0 cost of no uncertainty model policy)

        Returns
        -------
        risk_premium: float
            the risk premium of the recourse policy
        """

        # try to import a no uncertainty version with the same calibration; if one does not exist,
        # make one and solve it
        cwd = os.getcwd()
        no_unc_path = ''.join([cwd, '/data/output/', self.cal, "_inv_output.nc"])
        
        # try to open no uncertainty model results
        try:
            ds_no_unc = xr.open_dataset(no_unc_path)

        # if they don't exist, make your own
        except FileNotFoundError:
            m_no_unc = INVBaseModel(self.cal)
            m_no_unc.solve_opt(save_output=True)
            ds_no_unc = xr.open_dataset(no_unc_path)

        # find total costs for the policy
        total_cost_no_unc = (0.5 * ds_no_unc.cbars * ds_no_unc.investment**2).sum('sector')
        
        # extract only the t = 0 cost
        t0_total_cost_no_unc = total_cost_no_unc.values[0]

        # find total costs (in first period, at least) of the recourse policy
        total_cost_rec = 0.5 * self.cbars.value @ self.x_proc[0][0]**2

        # extract only the t = 0 cost
        t0_total_cost_rec = total_cost_rec[0]

        # risk premium is the difference between the no uncertainty case and the recourse case
        risk_premium = (t0_total_cost_rec - t0_total_cost_no_unc)/(t0_total_cost_no_unc)
        return risk_premium * 100
