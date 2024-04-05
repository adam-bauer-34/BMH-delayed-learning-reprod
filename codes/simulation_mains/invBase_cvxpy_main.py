"""7-Sector Abatement Investment Model using CVXPY.

The abatement investment cost functions in this model are calibrated
to be in line with the IPCC's AR6.

Adam Michael Bauer
University of Illinois Urbana-Champaign
World Bank Group
9.11.2023

To run: python invBase_cvxpy_main.py [cal] [solve_model] [save_output]
"""

import sys

from src.invBase_model import INVBaseModel

# parse command line inputs
cal = sys.argv[1]
solve_model = int(sys.argv[2])
save_output = int(sys.argv[3])

# instantiate model class
i = INVBaseModel(cal)

# run model
if solve_model:
    i.solve_opt(save_output=save_output)