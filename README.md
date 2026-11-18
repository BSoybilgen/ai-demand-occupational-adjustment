# When AI Automates Knowledge Work: Aggregate Demand and Occupational Adjustment

Code and data for the working paper by Baris Soybilgen (18 November 2026). The paper is in
[`paper/main_wp.pdf`](paper/main_wp.pdf).

The folder holds the model, the inputs it reads, and the saved results of every model run that the
paper uses. With these you can

1. rebuild the tables, the figures and the numbers quoted in the text from the saved results
   (about two minutes), and
2. solve the model again from the inputs (the main experiment takes about a quarter of an hour; all runs
   together take many hours).

## Contents

| Path | What it is |
|---|---|
| `paper/main_wp.pdf` | The working paper. |
| `capability_model/` | The model and the reporting code (Python). |
| `data/household_calibration/` | The three input files of the model (see "Data" below). |
| `results/` | Saved results of the model runs used in the paper. |
| `output/tables/`, `output/figures/` | Tables, numbers and figures written by the report. They are included as produced, and the report rewrites them. |
| `run_all.sh` | The command for every model run, in the order of the paper. |
| `requirements.txt` | Python packages. |
| `LICENSE`, `CITATION.cff` | Licence of the code, data and results; how to cite. |

## Requirements

Python 3 with `numpy` and `matplotlib` (tested with Python 3.15.0, numpy 2.5.4, matplotlib 3.11.2).

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

All commands are run from the root of this folder.

## 1. Tables, figures and numbers from the saved results

```sh
python -m capability_model.report
```

The first run computes the reference equilibria and keeps them in `results/stationary_cache/`;
it takes about two minutes. Later runs take about half a minute. No transition path is solved.

The report writes:

| Paper | File |
|---|---|
| Table 2, the reference and the data | `output/tables/fit.tex` |
| Table 3, capacity, spending and output | `output/tables/aggregates.tex` |
| Table 4, employment, labor services and pay | `output/tables/adjustment.tex` |
| Table 5, the labor market | `output/tables/labor.tex` |
| Table 6, the initial knowledge cohort | `output/tables/cohort.tex` |
| Table 7, the cohort by qualification and liquid assets | `output/tables/groups.tex` |
| Table 8, welfare in the main experiment | `output/tables/welfaremain.tex` |
| Table 9, two policy experiments | `output/tables/experiments.tex` |
| Table 10, the spending rule and access to manual work | `output/tables/robust.tex` |
| Figure 1 | `output/figures/aggregates.pdf` |
| Figure 2 | `output/figures/labor.pdf` |
| Figure 3 | `output/figures/cohort.pdf` |
| Figure 4 | `output/figures/experiments.pdf` |
| Every number quoted in the text | `output/tables/numbers.tex` |

The tables are LaTeX fragments; they contain cross-references to labels of the paper.
`numbers.tex` defines one LaTeX command for each number that the text of the paper quotes, so that
no number in the text is typed by hand. The report also writes further tables (`regimes`, `access`,
`spending`, `rentshare`, `ownership`, `wagespeed`, `welfare`, `later`, `boom`, `limit`, `checks`,
`applications`, `anchor`, `accesswelfare`, `parameters`) and one further figure (`regimes.pdf`) with
more detail on the same runs. The paper does not print them.

Table 1 (parameters) and Table 11 (occupations) are not produced by this code: Table 1 lists the
parameter values, which are in `capability_model/parameters.py` and `data/household_calibration/`,
and Table 11 is computed from the survey data (see "Data").

## 2. Solving the model again

The main experiment of Section 6:

```sh
python -m capability_model.run transition
```

It writes `results/transition_self-financed_f35.npz` (the paths), `results/transition.json`
(settings and convergence) and `results/welfare.npz` (the welfare measures), and replaces the saved
files of the same name. Add `--tag _name` to write under another name.

`run_all.sh` lists the command for every run. Each result file carries the suffix given by `--tag`:

| Suffix | Run | Used in |
|---|---|---|
| (none) | Main experiment | Section 6 |
| `_cut_main` | Interest rate responds to unused capacity | Section 7.1 |
| `_cut_foresight`, `_cut_mpc25`, `_cut_mpc75` | Three tests of the boom | Section 7.1, footnote |
| `_transfer50` | Transfer of half the rise in licence rents | Section 7.2 |
| `_transfer100` | Transfer of the whole rise | Section 7.2, footnote |
| `_anchor05` | Benchmark: the spending anchor follows income | Section 8.1 |
| `_mpc75` | Capitalists spend three quarters of an increase in income | Section 8.1 |
| `_access`, `_access_anchor05` | Broader access to manual work, under each spending rule | Section 8.2 |
| `_access_noai`, `_access_anchor05_noai` | Broader access with AI capability at its initial level | Section 8.2 |
| `_rent36` | Licence rent of 36 percent of the price of AI capital | Section 8.3 |
| `_own30` | Residents hold 30 percent of payout claims | Section 8.3 |
| `_wage10` | Posted wages close 10 percent of the gap each quarter | Section 8.3 |
| `_purchases` | Limit case with public purchases | Section 6.3, footnote |
| `_eta10`, `_ai30` | Elasticity of one between tasks; AI 30 percent cheaper than labor | Section 5.2, footnote |
| `_noapp…`, `_rho80…`, `_rho100…`, `_acc25` | Other assumptions on the entry into manual work | Sections 5.3 and 8.2, footnotes |

`python -m capability_model.run transition --help` describes every option.

**What has been checked.** On 18 November 2026 the report was run from this folder; its 25 files in
`output/tables/` are identical to the files from which the paper was built, and the four figures
match. The main experiment was solved again from this folder with the command above; the new result is identical to the saved one in each of its 47 arrays
and in the welfare file, after the same 212 passes.
The other runs were not solved again from this folder. Their commands in `run_all.sh` are those
recorded when the saved results were produced.

**Run times.** One pass over a path of 160 quarters takes about 4 seconds on the
author's laptop. `results/transition*.json` records the number of passes that each run needed:
between about 30 and 500 (212 for the main experiment). The runs with a moving spending anchor,
with the interest-rate rule and with public purchases are the longest.

## Data

`data/household_calibration/` holds the three files that the model reads. They come from the
calibration of an earlier version of the model to the U.S. Survey of Income and Program
Participation (public-use files of 2023, 2024 and 2025; Census Bureau). As Section 5 of the paper
explains, these household parameters are carried over and are not estimated again here.

| File | Content |
|---|---|
| `parameters.json` | Household parameters: worker types and their masses, efficiency units, discount factors, receipts by status, flows into and out of the private workforce, matching and search parameters, training costs, renewal. |
| `initial.npz` | The asset grid and the stationary distribution of the earlier calibration. The model uses them for the grid of liquid assets and for the share of each type that is qualified for the other occupation. |
| `reference.json` | Aggregates of the earlier reference; the model uses its value added as the level that the new reference reproduces. |

The data moments in the column "Data" of Table 2 are in `capability_model/parameters.py`
(`TARGETS`). The values chosen for this paper (the path of AI capability, the cost of AI, the
licence rent, the spending share of capitalists, the speed of wages and others) are in the same
file, each with its source.

Not included: the survey microdata, which are available from the U.S. Census Bureau, and the code
of the earlier household calibration. The statistics of Appendix C and Table 11 are computed from
the microdata and cannot be rebuilt from this folder.

## The code

| File | Content |
|---|---|
| `production.py` | Technology, task assignment, factor demand, supply of AI capital, investment (Section 3). |
| `households.py` | Values and choices of residents, search and matching, status flows, job values (Section 2, Appendix A). |
| `stationary.py` | The reference equilibrium and its calibration (Sections 4.7 and 5). |
| `transition.py` | Transition paths, including the yearly steps that are not foreseen (Section 4.7, Appendix B.2). |
| `welfare.py` | Welfare measures (Appendix B.3). |
| `parameters.py` | Parameters and the reading of the input files. |
| `run.py` | Command line. |
| `report.py` | Tables, figures and numbers. |
| `wording.py` | The vocabulary of the paper (see below). |

**Vocabulary.** The code was written before the final wording of the paper, and its names and
comments use older terms. `wording.py` translates them when the tables and figures are written.

| In the code | In the paper |
|---|---|
| return on claims | real interest rate |
| capability | AI capability |
| instruction | training |
| easier / unchanged access | broader / baseline access |
| anchor fixed, "the experiment" | main experiment |
| anchor follows income, second state of demand | benchmark |
| employment account | employer |
| liquid claims | liquid assets |

## Licence and citation

The code, the input files in `data/` and the results in `results/` and `output/` are licensed under
the [Creative Commons Attribution-NonCommercial 4.0 International licence](LICENSE) (CC BY-NC 4.0).
You may use, share and adapt them for non-commercial purposes, provided that you give credit.
Commercial use is not permitted.

To give credit, cite the paper:

> Soybilgen, B. (2026). *When AI Automates Knowledge Work: Aggregate Demand and Occupational
> Adjustment*. Working paper, 18 November 2026.

The paper in `paper/main_wp.pdf` is not covered by this licence. It is © Baris Soybilgen and is
distributed under the terms stated on its SSRN page.
