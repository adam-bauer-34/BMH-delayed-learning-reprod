# Data Availability Statement for "How delayed learning about climate uncertainty impacts decarbonization investment strategies"

By: Adam Michael Bauer -- adammb4 [at] illinois [dot] edu

This directory contains a set of codes that reproduces all of the figures and analysis carried out in: *How delayed learning about climate uncertainty impacts decarbonization investment strategies* by Bauer, McIsaac, and Hallegatte. 

To cite our working paper that uses these codes: [Bauer, A. M., F. McIsaac, S. Hallegatte. *How Delayed Learning about Climate Uncertainty Impacts Decarbonization Investment Strategies*. World Bank Policy Research Working Paper No. WPS10743, World Bank Group, Washington DC, 2024.](https://documents.worldbank.org/en/publication/documents-reports/documentdetail/099829103282438373/idu1f2d86d77127091490d1a6df1dc342f15d10b)

All of the data used in our study is taken and/or interpreted from publically available publications and reports. Individual numbers used in the simulations can be found in the `with-gurobi/data/cal/` files for each simulation. (An equivalent folder exists for the `without-gurobi` codes.) All values are taken from the following papers or reports:

- Data for the remaining carbon budget and its uncertainty is taken from [Dvorak *et al.*, 2022](https://www.nature.com/articles/s41558-022-01372-y), see Table 2, the "No cessation" rows.
- Marginal abatement costs and abatement potentials in each economic sector we considered is taken from the [Intergovernmental Panel on Climate Change's Sixth Assessment Report, specifically the contributions of Working Group III, Figure SPM.7 on p. 38 of the *Summary for Policymakers*](https://www.ipcc.ch/report/ar6/wg3/downloads/report/IPCC_AR6_WGIII_SummaryForPolicymakers.pdf). An excel spreadsheet of this data was provided to AMB by one of the authors of the IPCC Report, and is available upon request.
- Capital depreciation rates are taken from [Philibert, C., 2007](https://www.osti.gov/etdeweb/biblio/20962174), see Figure 8. The capital depreciation rate is the inverse of the capital lifetime.
- The social discount rate is taken from [Drupp *et al.*, 2018](https://www.aeaweb.org/articles?id=10.1257/pol.20160240), their median estimate.

Last edited: 29 March, 2024.
