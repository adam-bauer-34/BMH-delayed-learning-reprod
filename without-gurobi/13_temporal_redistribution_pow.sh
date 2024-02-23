#!/bin/bash

echo 'Carrying out simulations...'
echo 'Base models...'

python simulation_mains/invBase_cvxpy_main.py ar6pow_15 1 1
python simulation_mains/invBase_cvxpy_main.py ar6pow_17 1 1
python simulation_mains/invBase_cvxpy_main.py ar6pow_2 1 1

python simulation_mains/macBase_cvxpy_main.py ar6pow_15 1 1
python simulation_mains/macBase_cvxpy_main.py ar6pow_17 1 1
python simulation_mains/macBase_cvxpy_main.py ar6pow_2 1 1

echo 'Models with uncertainty...'

python simulation_mains/invRec_RiskPrem_cvxpy_main.py ar6pow_15 N1_T40_B6 6 1 1
python simulation_mains/invRec_RiskPrem_cvxpy_main.py ar6pow_17 N1_T40_B6 6 1 1
python simulation_mains/invRec_RiskPrem_cvxpy_main.py ar6pow_2 N1_T40_B6 6 1 1

python simulation_mains/macRec_RiskPrem_cvxpy_main.py ar6pow_15 N1_T40_B6 6 1 1
python simulation_mains/macRec_RiskPrem_cvxpy_main.py ar6pow_17 N1_T40_B6 6 1 1
python simulation_mains/macRec_RiskPrem_cvxpy_main.py ar6pow_2 N1_T40_B6 6 1 1

echo 'Making Figure 13...'

python figure_mains/temporal_redist.py pow