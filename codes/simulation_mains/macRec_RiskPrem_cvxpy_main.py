"""Marginal abatement cost curve model with recourse main file.

Adam Michael Bauer
University of Illinois Urbana-Champaign
World Bank Group
10.2.2023

To run: python macRec_RiskPrem_cvxpy_main.py [cal] [rec_cal] [N_samples] [method] [save_output]
"""

import sys
import os

import numpy as np
import matplotlib.pyplot as plt

from src.macRec_model import MACRecourseModel
from src.macRecExp_model import MACRecourseModelExp
from datatree import DataTree

# get rid of stupid future warnings stuff, can delete in the future if necessary (lol)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

cal = sys.argv[1]
rec_cal = sys.argv[2]
N_samples = int(sys.argv[3])
method =int(sys.argv[4])
save_output = int(sys.argv[5])

dt = 5.0
learning_times = np.arange(0.0, 80.0, dt)

data_tree_dict = {}
for Tstar in learning_times:

    print("------------------------------")
    print("WE ARE ON T*={}.".format(Tstar))
    print("------------------------------")
    # reset seed so every time we initiate the tree we get the same RCB distribution
    np.random.seed(9324)

    #temporary model class
    # if Tstar = 0, run the expectation model, and if Tstar > 0, run the recourse model
    if Tstar == 0:
        tmp_m = MACRecourseModelExp(cal, N_samples, method)

    else:
        tmp_m = MACRecourseModel(cal, rec_cal)

        # initialize tree structure
        tmp_m.tree_init(print_outcome=True)

        tmp_m.rev_times = [Tstar]
        tmp_m.period_times = [0.0, Tstar, tmp_m.T.value]

    # initialize problem variables and expressions
    tmp_m.prob_init(print_outcome=False)

    # solve problem
    tmp_m.solve_opt(save_output=False, suppress_prints=True)

    # add data tree to dict
    data_tree_dict[str(Tstar)] = tmp_m.data_tree

cwd = os.getcwd()
path_to_data = '/data/output/'

# make datatree of each simulation
data_tree = DataTree.from_dict(data_tree_dict, 'learning_time')
dt_path = ''.join([cwd, path_to_data, cal, '_', rec_cal, '_method', str(method), '_mac_rp_data.nc'])

if save_output:
    data_tree.to_netcdf(filepath=dt_path, mode='w', format='NETCDF4', engine='netcdf4')

    print("\nData from each simulation saved to:\n", dt_path)

else:
    print(data_tree)