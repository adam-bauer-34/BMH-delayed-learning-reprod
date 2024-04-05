"""Abatement investment model with recourse main file.

Adam Michael Bauer
University of Illinois Urbana-Champaign
World Bank Group
10.5.2023

To run: python invRecEmis_cvxpy_main.py [cal] [N_samples] [method] [solve_model] [save_output]
"""

import sys

from src.macRecExpEmis_model import MACRecourseModelExpEmis

cal = sys.argv[1]
N_samples = int(sys.argv[2])
method = int(sys.argv[3])
solve_model = int(sys.argv[4])
save_output = int(sys.argv[5])

m = MACRecourseModelExpEmis(cal, N_samples, method)

# initialize problem variables and expressions
m.prob_init(print_outcome=False)

# if you're interested, solve the model
if solve_model:
    m.solve_opt(save_output=save_output)