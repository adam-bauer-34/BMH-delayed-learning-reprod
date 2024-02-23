"""Abatement investment model with recourse main file.

Adam Michael Bauer
University of Illinois Urbana-Champaign
World Bank Group
10.5.2023

To run: python invRecEmis_cvxpy_main.py [cal] [rec_cal] [solve_model] [save_output]
"""

import sys

from src.macRecEmis_model import MACRecourseModelEmis

cal = sys.argv[1]
rec_cal = sys.argv[2]
solve_model = int(sys.argv[3])
save_output = int(sys.argv[4])

m = MACRecourseModelEmis(cal, rec_cal)

# initialize tree structure
m.tree_init(print_outcome=True)

# initialize problem variables and expressions
m.prob_init(print_outcome=False)

# if you're interested, solve the model
if solve_model:
    m.solve_opt(save_output=save_output)