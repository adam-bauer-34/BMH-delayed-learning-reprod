#!/bin/bash

echo 'Carrying out simulations...'
echo 'Base models...'

python simulation_mains/invBase_cvxpy_main.py ar6_17 1 1
python simulation_mains/invBase_cvxpy_main.py ar6_2 1 1

python simulation_mains/macBase_cvxpy_main.py ar6_17 1 1
python simulation_mains/macBase_cvxpy_main.py ar6_2 1 1

echo 'Models with uncertainty...'

python simulation_mains/invRec_RiskPrem_cvxpy_main.py ar6_17 N1_T30_B8 8 3 1
python simulation_mains/invRec_RiskPrem_cvxpy_main.py ar6_2 N1_T30_B8 8 3 1

python simulation_mains/macRec_RiskPrem_cvxpy_main.py ar6_17 N1_T30_B8 8 3 1
python simulation_mains/macRec_RiskPrem_cvxpy_main.py ar6_2 N1_T30_B8 8 3 1

echo 'Making Figure 2...'

python figure_mains/effect_of_learning.py low-linear
