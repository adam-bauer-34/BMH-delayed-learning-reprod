# How delayed learning about climate uncertainty impacts decarbonization investment strategies reproduction without GUROBI

This set of codes recreates all the figures from *How delayed learning about climate uncertainty impacts decarbonization investment strategies* using the default CVXPY solver.

Simply run the numbered code to recreate the desired figure. You may need to execute:
```
    chmod +x script_name
```
to grant execution permissions (hence the `+x`) to the script you want to run.

All figures will be deposited into the `figs` folder. To run indiviudal simulations, you can run any of the files in `simulation_mains`, and to make individual figures, you can run any file in the `figure_mains` folder. **Note** you should run all scripts from this directory (the one that contains this README file). As an example, let's say you want to run the `invBase_cvxpy_main.py` file in the `ar6_15` calibration, but not save the output. Then in your command line, you'd use:
```
python simulation_mains/invBase_cvxpy_main.py ar6_15 1 0
```

**Note:** You should be operating in the Python environment provided at the head directory. Without it, I make no guarantees any of this will run on your machine (and even then, well, mileage may vary...).