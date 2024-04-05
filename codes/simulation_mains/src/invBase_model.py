"""Generic Abatement Investment model class.

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

from scipy.optimize import root
from .macBase_model import MACBaseModel

class INVBaseModel():
    """Abatement investment model class.

    This class contains the necessary attributes and methods for solving the
    abatement investment model (INV model) using `cvxpy`.

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

    def __init__(self, cal):
        self.cal = cal # calibration name

        # if the calibration name is not a string, throw an error
        if not isinstance(self.cal, str):
            raise ValueError("Calibration name must be a string. You passed {}, which is type {}.".format(self.cal, type(self.cal)))

        self._import_calibration()

        # big ol print message
        print("Model import successful. This is the abatement investment model.")
        print("\n-------------------\nGLOBAL PARAMETERS\n-------------------\n")
        print("Calibration name: {}.".format(self.cal))
        print("Time horizon: {} yrs.".format(self.T.value))
        print("Social discount rate: {}%/yr.".format(self.r.value*100))
        print("Remaining carbon budget: {} GtCO2.".format(self.B.value))
        print("Time discretization (dt): {} yrs.".format(self.dt.value))
        print("\n-------------------\nSECTORAL PARAMETERS\n-------------------\n")
        print(''.join(["SECTOR: {}. EMISSIONS RATE: {} GtCO2/yr. DEPRECIATION RATE: {}%/yr.  COST COEFFICIENT: {} ($/tCO2) / (GtCO2/yr^3). POWER: {}.\n".format(i,j,k,l,m) for i,j,k,l,m in zip(self.sectors, self.abars.value, self.deltas.value*100, self.cbars.value, self.powers)]))

    def _import_calibration(self):
        """Import calibration.

        See the TAXONOMY. The INVModel class is passed a string for which calibration
        of the model we're using, and this function takes that string and calibrates
        the INV model based on the MAC model.

        If no INV model calibration exists, we make one using _cal_model.
        """

        # get current working directory and make two paths to data
        cwd = os.getcwd()
        path_glob = ''.join([cwd, '/data/cal/', self.cal, '_glob.csv'])
        path_secs = ''.join([cwd, '/data/cal/', self.cal, '_secs.csv'])

        # import csvs using Pandas
        df_glob = pd.read_csv(path_glob, ',', header=0)
        df_secs = pd.read_csv(path_secs, ',', header=0, index_col=0)

        # parse global parameters
        self.r = cp.Parameter(nonneg=True, 
                              value=df_glob['r'].values[0]) # social discount rate
        self.beta = (1+self.r)**(-1) # discount factor 
        self.B = cp.Parameter(nonneg=True,
                              value=df_glob['B'].values[0]) # remaining carbon budget
        self.T = cp.Parameter(nonneg=True, 
                              value=df_glob['T'].values[0]) # terminal time horizon
        self.dt = cp.Parameter(nonneg=True,
                               value=df_glob['dt'].values[0]) # time discretization
        self.times = np.arange(0.0, self.T.value, self.dt.value) # times we're solving the optimization problem at
        self.betas = self.beta.value**self.times # vector of discount factors
        self.psi_0 = cp.Parameter(nonneg=True, 
                                  value=df_glob['psi_0'].values[0]) # initial conditions for cumulative emissions
        
        # parse sector-specific parameters
        self.sectors = df_secs.columns.values # sectors
        self.N_secs = len(self.sectors) # number of sectors 
        self.abars = cp.Parameter(self.N_secs, nonneg=True, 
                                  value=df_secs.loc['abar'].values) # emissions rates by sector
        self.deltas = cp.Parameter(self.N_secs, nonneg=True, 
                                   value=df_secs.loc['delta'].values) # capital depreciations rate in each sector
        self.a_0s = cp.Parameter(self.N_secs, nonneg=True, 
                                 value=df_secs.loc['a_0'].values) # initial amount of abatement in each sector
        # NOTE: gbars is not a `cvxpy` parameter since we don't use it in the optimization in INV model
        self.gbars = df_secs.loc['gbar'].values # marginal abatement cost
        self.powers = df_secs.loc['power'].values # power for marginal abatement cost curves
        
        # try to import cbar parameters. if none exist, make them
        # NOTE: the KeyError comes from the fact that df_secs.loc['cbar'] has no entries
        try:
            self.cbars = cp.Parameter(self.N_secs, nonneg=True, 
                                      value=df_secs.loc['cbar'].values/1000.)

        except KeyError:
            # make relative costs for calibration
            self.rel_costs = np.array([self.gbars[i]/self.gbars[0] for i in range(self.N_secs)])
            self._cal_model(save_cal=True)
    
    def _cal_model(self, save_cal):
        """Calibrate abatement investment model.

        The condition is that the total costs for the MAC model with
        equivalent conditions have the same costs.

        Parameters
        ----------
        save_cal: bool
            save calibration to csv?
        """

        # instance the MAC model using INV model data
        m = MACBaseModel(self.cal)

        print("Solving MAC model...")
        m.solve_opt(save_output=False, verbose=False, cal_purposes=True)

        # total costs for MAC model
        mac_total_cost = m.prob.value

        print("Now we're calibrating the INV model...")
        # now find cbars such that the total cost is equal for inv and mac models
        sol = root(self._get_diff, x0=10000.0, args=(mac_total_cost))
        
        print("Calibration complete.")
        cbars = sol.x[0] * self.rel_costs

        print("The optimal cost parameters are:")
        print(''.join(["SECTOR: {}.    COST COEFFICIENT: {}.\n".format(i,j) for i,j in zip(self.sectors, cbars)]))
        
        self.cbars = cp.Parameter(self.N_secs, nonneg=True, value=cbars/1000.) # abatement investment cost by sector

        if save_cal:
            df = pd.DataFrame([self.gbars, self.abars.value, self.deltas.value, self.a_0s.value, self.powers, self.cbars.value * 1000.],
                              index=['gbar', 'abar', 'delta', 'a_0', 'power', 'cbar'], 
                              columns=self.sectors)

            cwd = os.getcwd()
            path = ''.join([cwd, '/data/cal/', self.cal, '_secs.csv'])
            df.to_csv(path)
        
    def _get_diff(self, ci, mac_total_cost):
        """Get difference between INV and MAC model total costs.

        This function is used by a root finder to calibrate the INV model.

        Parameters
        ----------
        ci: float
            the missing cost parameter; overall cost factor

        mac_total_cost: float
            total cost of equivalent MAC policy

        Returns
        -------
        diff: float
            difference between INV policy with ci and MAC model
        """

        tmp_cbars = ci * self.rel_costs

        self.cbars = cp.Parameter(self.N_secs, nonneg=True, value=tmp_cbars / 1000.)

        self.solve_opt(save_output=False, verbose=False, cal_purposes=True)

        diff = mac_total_cost - self.prob.value * 1000.

        return diff

    def solve_opt(self, save_output=True, verbose=True, cal_purposes=False):
        """Solve optimization problem with `cvxpy`.

        NOTE: the last two arguments only affect the print statements that come along
        with the optimization.

        Parameters
        ----------
        save_output: bool (default=True)
            save output of optimization to netCDF?

        verbose: bool (default=True)
            print verbose output from optimization?

        cal_purposes: bool (default=False)
            are we using this function to calibrate the model? 
        """

        # define variables; investment (x), abatement (a),
        # cumulative emissions (psi), and emissions flux (F_psi)
        self.x = cp.Variable((self.N_secs, len(self.times)), nonneg=True)
        self.a = cp.Variable((self.N_secs, len(self.times)+1), nonneg=True)
        self.psi = cp.Variable(len(self.times)+1, nonneg=True)
        self.F_psi = cp.Variable(len(self.times), nonneg=True)

        # generate cumulative emissions and right hand side of emissions flux
        self.psi_vec = self._get_psi_vector()
        self.flux_rhs_v = self._get_flux_rhs()

        # generate constraints on abatement in each sector
        self.a_vecs = []
        for i in range(self.N_secs):
            tmp_a_vec = self._get_abatement_vector(self.a_0s.value[i],
                                           self.deltas.value[i],
                                           self.x[i])
            self.a_vecs.append(tmp_a_vec)
        
        # make list of constraints
        self.constraints = []

        ## on cumulative emissions
        self.constraints.append(self.psi[0] == self.psi_0) # initial condition
        self.constraints.append(self.psi[1:] == self.psi_vec) # rest equal to flux
        self.constraints.append(self.psi <= self.B) # capped at RCB

        ## on abatement
        self.constraints.append(self.a[:, 0] == self.a_0s) # initial a is the IC
        self.constraints.extend([self.x[i, :] - self.deltas[i] * self.a[i, :-1] >= 0 for i in range(self.N_secs)]) # irreversibility constraints
        self.constraints.extend([self.a[i, 1:] == self.a_vecs[i] for i in range(self.N_secs)]) # rest of values equal to fluxes
        self.constraints.extend([self.a[i,:] <= self.abars[i] for i in range(self.N_secs)]) # cap abatement at emissions rate

        ## on flux
        self.constraints.append(self.F_psi == self.flux_rhs_v) # flux equality for SCC

        # define and solve problem
        obj_term = cp.sum([self.cbars[i].value * (self.powers[i] + 1)**(-1) * self.x[i]**(self.powers[i] + 1) for i in range(self.N_secs)])
        self.prob = cp.Problem(cp.Minimize(self.betas @ obj_term), self.constraints)
        self.prob.solve(solver=cp.GUROBI, verbose=verbose)

        # get statistics from solver 
        self.solve_stats = self.prob.solver_stats

        # if we save the output, save it (so long as the optimizer returned the optimal result)
        if save_output and self.prob.status == 'optimal':
            self._save_output()

        # if solving for calibration purposes, just confirm that we exited with optimal status
        elif cal_purposes and self.prob.status == 'optimal':
            print("INV model finished with optimal status.")

        # else, print the outcome of the optimization
        else:
            print("Status: ", self.prob.status)
            print("Total cost: ", self.prob.value)
            print("Investment path: ", self.x.value)
            print("Abatemnet: ", self.a.value)
            print("Cumulative emissions: ", self.psi.value)

            # the SCC is the final condition value
            print("Time zero SCC: ", -1*self.constraints[-1].dual_value[0] * 1000.)
        
    def _save_output(self):
        """Save output of optimization to netCDF file.
        """

        # make xarray dataset object
        ds = xr.Dataset(data_vars={'investment': (['sector', 'time'], self.x.value),
                        'abatement': (['sector', 'time_state'], self.a.value),
                        'cumulative_emissions': (['time_state'], self.psi.value),
                        'scc': (['time'], self.constraints[-1].dual_value),
                        'cbars': (['sector'], self.cbars.value * 1000.),
                        'abars': (['sector'], self.abars.value),
                        'deltas': (['sector'], self.deltas.value),
                        'a_0s': (['sector'], self.a_0s.value),
                        'power': (['sector'], self.powers)},
                coords={'time': (['time'], self.times),
                        'time_state': (['time_state'], np.hstack([self.times, 1e10])),
                        'sector': (['sector'], self.sectors)},
                attrs={'total_cost': self.prob.value * 1000.,
                       'r': self.r.value,
                       'beta':self.beta.value,
                       'dt': self.dt.value,
                       'T': self.T.value,
                       'B': self.B.value,
                       'psi_0': self.psi_0.value,
                       'status': self.prob.status,
                       'solve_time': self.solve_stats.solve_time})

        # get current working directory, save output to data/output/ folder
        cwd = os.getcwd()
        path = ''.join([cwd, '/data/output/', self.cal, '_inv_output.nc'])
        ds.to_netcdf(path=path, mode='w', format='NETCDF4', engine='netcdf4')
        print("\n-------------------------------------\n")
        print("Data successfully saved to file:\n{}".format(path))
        print("\n-------------------------------------\n")

    def _get_psi_vector(self):
        """Get the vector of cumulative emissions for t > 0.

        This is essentially the cumulative emissions equation of motion evaluated
        at each point in time.

        Returns
        -------
        psi_init_piece + dt*flux: `cvxpy.Expression`
            the cumulative emissions equation of motion
        """

        # piece associated with initial conditions
        psi_init_piece = np.ones(len(self.times)) * self.psi_0.value

        # piece associated with emissions flux
        flux_piece = np.tri(len(self.times)) @ self.F_psi

        # return equation of motion
        return psi_init_piece + self.dt.value * flux_piece
    
    def _get_flux_rhs(self):
        """Get the emissions flux. 

        Necessary to get SCC from optimizer. 
        flux = \sum_{sectors}{emis - abatement} for all t

        Returns
        -------
        flux: `cvxpy.Variable`
            emissions flux
        """

        # emissions flux
        # NOTE: only use abatements for t < T for flux 
        flux = cp.sum(self.abars) * np.ones(len(self.times)) - cp.sum(self.a[:, :-1], axis=0)
        return flux

    def _get_abatement_vector(self, a_0, delta, x_it):
        """Get abatement vector for a given sector.

        Parameters
        ----------
        a_0: float
            initial condition of abatement

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
        init_piece = (1.0 - delta * self.dt.value)**self.times * a_0

        # piece associated with investment (i.e., the abatement flux)
        delta_matrix = self._get_delta_matrix(delta)
        flux_piece = delta_matrix @ x_it

        # sum & return
        a_it = init_piece + self.dt.value * flux_piece
        return a_it

    def _get_delta_matrix(self, delta):
        """Generate matrix of capital deprecation values for abatement
        investment.

        Parameters
        ----------
        delta: float
            abatement capital depreciation rate

        Returns
        -------
        delta_matrix: (len(times), len(times)) matrix
            matrix of capital depreciation rates
        """

        delta_matrix = np.zeros((len(self.times), len(self.times)))
        for i in range(len(self.times)):
            for j in range(len(self.times)):
                if i == j:
                    delta_matrix[i,j] = 1.0
                elif j < i:
                    delta_matrix[i,j] = (1 - delta * self.dt.value)**(i-j)
                else:
                    continue
        return delta_matrix