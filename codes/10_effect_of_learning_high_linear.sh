#!/bin/bash

echo 'Carrying out simulations...'
echo 'Base models...'

python simulation_mains/invBase_cvxpy_main.py ar6hi_15 1 1
python simulation_mains/invBase_cvxpy_main.py ar6hi_17 1 1
python simulation_mains/invBase_cvxpy_main.py ar6hi_2 1 1

python simulation_mains/macBase_cvxpy_main.py ar6hi_15 1 1
python simulation_mains/macBase_cvxpy_main.py ar6hi_17 1 1
python simulation_mains/macBase_cvxpy_main.py ar6hi_2 1 1

echo 'Models with uncertainty...'

python simulation_mains/invRec_RiskPrem_cvxpy_main.py ar6hi_15 N1_T40_B6 6 1 1
python simulation_mains/invRec_RiskPrem_cvxpy_main.py ar6hi_17 N1_T40_B6 6 1 1
python simulation_mains/invRec_RiskPrem_cvxpy_main.py ar6hi_2 N1_T40_B6 6 1 1

python simulation_mains/macRec_RiskPrem_cvxpy_main.py ar6hi_15 N1_T40_B6 6 1 1
python simulation_mains/macRec_RiskPrem_cvxpy_main.py ar6hi_17 N1_T40_B6 6 1 1
python simulation_mains/macRec_RiskPrem_cvxpy_main.py ar6hi_2 N1_T40_B6 6 1 1

echo 'Making Figure 10...'

python figure_mains/effect_of_learning.py high-linear