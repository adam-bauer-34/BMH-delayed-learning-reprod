#!/bin/bash

echo 'Carrying out simulations...'
echo 'Base models...'

python simulation_mains/invBaseEmis_cvxpy_main.py ar6emis_15 1 1
python simulation_mains/invBaseEmis_cvxpy_main.py ar6emis_17 1 1
python simulation_mains/invBaseEmis_cvxpy_main.py ar6emis_2 1 1

python simulation_mains/macBaseEmis_cvxpy_main.py ar6emis_15 1 1
python simulation_mains/macBaseEmis_cvxpy_main.py ar6emis_17 1 1
python simulation_mains/macBaseEmis_cvxpy_main.py ar6emis_2 1 1

echo 'Models with uncertainty...'

python simulation_mains/invRecEmis_RiskPrem_cvxpy_main.py ar6emis_15 N1_T40_B6 6 1 1
python simulation_mains/invRecEmis_RiskPrem_cvxpy_main.py ar6emis_17 N1_T40_B6 6 1 1
python simulation_mains/invRecEmis_RiskPrem_cvxpy_main.py ar6emis_2 N1_T40_B6 6 1 1

python simulation_mains/macRecEmis_RiskPrem_cvxpy_main.py ar6emis_15 N1_T40_B6 6 1 1
python simulation_mains/macRecEmis_RiskPrem_cvxpy_main.py ar6emis_17 N1_T40_B6 6 1 1
python simulation_mains/macRecEmis_RiskPrem_cvxpy_main.py ar6emis_2 N1_T40_B6 6 1 1

echo 'Making Figure 8...'

python figure_mains/effect_of_learning.py emis