"""Generic Marginal Abatement Cost model class.

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

class MACBaseModel():
    """Marginal abatement cost curve model class.

    This class contains the necessary attributes and methods for solving the
    marginal abatement cost curve model (MAC model) using `cvxpy`.

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

    psi_0: `cvxpy.Parameter`, float
        initial amount of cumulative emissions
    
    a: `cvxpy.Variable`, (N_secs, len(times))
        abatement rate (decision variable in optimization)

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
        imports calibration based off of self.cal parameter
        
    solve_opt:
        solves optimization problem with specified parameters
    
    _save_output:
        saves output of optimization to netCDF

    _get_flux_rhs:
        get right hand side of flux for cumulative emissions equation of motion

    _get_psi_vector:
        generate the vector of cumulative emissions for t>0
    """

    def __init__(self, cal):
        self.cal = cal # calibration name

        # if the calibration name is not a string, throw an error
        if not isinstance(self.cal, str):
            raise ValueError("Calibration name must be a string. You passed {}, which is type {}.".format(self.cal, type(self.cal)))

        self._import_calibration()

        # big ol print message
        print("Model import successful. This is the marginal abatement cost curve model.")
        print("\n-------------------\nGLOBAL PARAMETERS\n-------------------\n")
        print("Calibration name: {}.".format(self.cal))
        print("Time horizon: {} yrs.".format(self.T.value))
        print("Social discount rate: {}%/yr.".format(self.r.value*100))
        print("Remaining carbon budget: {} GtCO2.".format(self.B.value))
        print("Time discretization (dt): {} yrs.".format(self.dt.value))
        print("\n-------------------\nSECTORAL PARAMETERS\n-------------------\n")
        print(''.join(["SECTOR: {}. EMISSIONS RATE: {} GtCO2/yr. MARGINAL ABATEMENT COST: {} ($/tCO2) / (GtCO2/yr). POWER: {}.\n".format(i,j,k,l) for i,j,k,l in zip(self.sectors, self.abars.value, self.gbars.value, self.powers)]))

    def _import_calibration(self):
        """Import calibration.

        See the TAXONOMY. The MACModel class is passed a string for which calibration
        of the model we're using.
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
        self.gbars = cp.Parameter(self.N_secs, nonneg=True, 
                                  value=df_secs.loc['gbar'].values) # marginal abatement cost
        self.powers = df_secs.loc['power'].values # power for marginal abatement cost curves

    def solve_opt(self, save_output=True, verbose=True, cal_purposes=False):
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
        """

        # define variables (abatement and emissions flux)
        self.a = cp.Variable((self.N_secs, len(self.times)), nonneg=True)
        self.F_psi = cp.Variable(len(self.times), nonneg=True)

        # generate cumulative emissions and right hand side of emissions flux
        self.psi_vec = self._get_psi_vector()
        self.flux_rhs_v = self._get_flux_rhs()
        
        # make list of constraints
        self.constraints = [self.a[i,:] <= self.abars[i] for i in range(self.N_secs)]
        self.constraints.append(self.psi_vec <= self.B)
        self.constraints.append(self.F_psi == self.flux_rhs_v)

        # define and solve problem
        obj_term = cp.sum([self.gbars[i].value * (self.powers[i] + 1)**(-1) * self.a[i]**(self.powers[i] + 1) for i in range(self.N_secs)])
        self.prob = cp.Problem(cp.Minimize(self.betas @ obj_term), self.constraints)
        self.prob.solve(solver=cp.GUROBI, verbose=verbose)

        # get statistics from solver 
        self.solve_stats = self.prob.solver_stats

        # if we save the output, save it (so long as the optimizer returned the optimal result)
        if save_output and self.prob.status == 'optimal':
            self._save_output()
        
        elif cal_purposes and self.prob.status=='optimal':
            print("MAC model finished with optimal status.")

        # else, print the outcome of the optimization
        else:
            print("Status: ", self.prob.status)
            print("Total cost: ", self.prob.value)
            print("Path: ", self.a.value)
            print("Cumulative emissions: ", self.psi_vec.value)

            # the SCC is the final condition value
            print("SCC: ", self.constraints[-1].dual_value[0])
        
    def _save_output(self):
        """Save output of optimization to netCDF file.
        """

        # make xarray dataset object
        ds = xr.Dataset(data_vars={'abatement': (['sector', 'time'], self.a.value),
                        'cumulative_emissions': (['time'], self.psi_vec.value),
                        'scc': (['time'], self.constraints[-1].dual_value),
                        'gbars': (['sector'], self.gbars.value),
                        'abars': (['sector'], self.abars.value),
                        'power': (['sector'], self.powers)},
                coords={'time': (['time'], self.times),
                        'sector': (['sector'], self.sectors)},
                attrs={'total_cost': self.prob.value,
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
        path = ''.join([cwd, '/data/output/', self.cal, '_mac_output.nc'])
        ds.to_netcdf(path=path, mode='w', format='NETCDF4', engine='netcdf4')
        print("Data successfully saved to file:\n{}".format(path))

    def _get_psi_vector(self):
        """Get the vector of cumulative emissions for t > 0.

        This is essentially the cumulative emissions equation of motion evaluated
        at each point in time.
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
        """

        # emissions flux
        flux = cp.sum(self.abars) * np.ones(len(self.times)) - cp.sum(self.a, axis=0)
        return flux