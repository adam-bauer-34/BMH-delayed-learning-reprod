"""Marginal Abatement Cost Model using CVXPY.

Adam Michael Bauer
University of Illinois Urbana-Champaign
World Bank Group
9.11.2023

To run: python macBase_cvxpy_main.py [cal] [solve_model] [save_output]
"""

import sys 

from src.macBase_model import MACBaseModel

# parse command line inputs
cal = sys.argv[1]
solve_model = int(sys.argv[2])
save_output = int(sys.argv[3])

# instantiate model class
m = MACBaseModel(cal)

# run model
if solve_model:
    m.solve_opt(save_output=save_output)