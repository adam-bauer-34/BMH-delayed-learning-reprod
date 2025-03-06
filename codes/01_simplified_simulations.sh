#!/bin/bash

echo 'Carrying out simulations...'
echo 'Base models...'

python simulation_mains/invBase_cvxpy_main.py simp 1 1

python simulation_mains/macBase_cvxpy_main.py simp 1 1

echo 'Models with uncertainty...'

python simulation_mains/invRec_cvxpy_main.py simp N1_T30_B3 1 1

python simulation_mains/macRec_cvxpy_main.py simp N1_T30_B3 1 1

echo 'Making Figure 1...'

python figure_mains/simp.py
