# A short running list of model names and their taxonomy.

Generic naming scheme for `cvxpy` programs: [model_type]\_[calc]\_cvxpy\_main.py

`model_type`: has the following values:
- `invBase`: The abatement investment model from Vogt-Schilb *et al.*, 2018 with no uncertainty.
- `macBase`: The marginal abatement cost curve model from Vogt-Schilb *et al.*, 2018 with no uncertainty.
- `invAdj`: The abatement investment model with an alternative parameterization of adjustment costs. No uncertainty.
- `macAdj`: The marginal abatement cost curve model with an alternative parameterization of adjustment costs. No uncertainty.
- `invRec`: The abatement investment model with recourse
- `macRec`: The marginal abatement cost curves model with recourse.

`calc`: has the following values:
- `RiskPrem`: Calculates the risk premium of the `model_type` against its no uncertainty counterpart.
**NOTE**: `calc` is an optional parameter. If `calc` is not in the script naming string, then that script simply solves a model with `model_type`. 

Naming schemes for economic calibration:
- `vs18`: The base case calibration from Vogt-Schlib *et al.*, 2018.
- `ar6_15`: The IPCC AR6-consistent calibration.
- `ar6_17`: IPCC AR6-consistent calibration with 1.7 deg C temperature goal.
- `ar6_2`: IPCC AR6-consistent calibration with 2 deg C temperature goal.
    - `[cal]_bs`: The above calibration with backstop technologies included.

Naming schemes for recourse calibration: `rec`\_N`num_rev`\_T`times`\_B`samples`
- `num_rev` is the number of times information is revealed, i.e., the "number of branching points" in the tree.
- `times` is the time(s) that the branching occurs after t = 0. Note for a 3-period model where the tree splits at t = 10 and t = 45, you'd write `times`=`1045`.
- `samples` is the number of samples you take from the RCB distribution in the final period.
    - My default for most of my runs is `samples`=200. But this can be changed (and will likely change for sensitivity tests).