#!/bin/sh
# Solves every model run that the paper uses and writes the results to results/.
# Run from the root of this folder:   sh run_all.sh
#
# The folder already contains these results, so this script is needed only to recompute them.
# Each run is independent of the others; a single line can be run on its own.
# Tables, figures and the numbers quoted in the text are then written by
#     python -m capability_model.report
#
# The names after --tag are the suffixes of the result files (README.md lists what each one is).

set -e
PY="${PYTHON:-python}"
RUN="$PY -m capability_model.run transition"

# ---- Section 6: the main experiment
$RUN

# ---- Section 8.1: capitalists' spending
# the benchmark, in which the spending anchor follows income
$RUN --anchor-speed 0.05 --memory 30 --tolerance 3e-4 --passes 500 --step-passes 400 --tag _anchor05
# capitalists spend three quarters of an increase in their income
$RUN --passes 300 --step-passes 200 --capitalist-mpc 0.75 --tag _mpc75

# ---- Section 8.2: broader access to manual work
$RUN --passes 300 --step-passes 200 --open-access --tag _access
$RUN --anchor-speed 0.05 --memory 30 --tolerance 3e-4 --passes 500 --step-passes 400 --open-access --tag _access_anchor05
# broader access with AI capability at its initial level (main spending rule, then benchmark)
$RUN --open-access --no-capability --passes 300 --tag _access_noai
$RUN --open-access --no-capability --anchor-speed 0.05 --memory 30 --tolerance 3e-4 --step-size 0.02 --passes 600 --tag _access_anchor05_noai

# ---- Section 8.3: licence rent, ownership of payouts, speed of wages
$RUN --passes 300 --step-passes 200 --rent-share 0.36 --tag _rent36
$RUN --passes 300 --step-passes 200 --payout-share 0.30 --tag _own30
$RUN --passes 300 --step-passes 200 --wage-speed 0.10 --tag _wage10

# ---- Section 7.1: the interest-rate response, and the three tests of the boom in its footnote
$RUN --rate-lower 0 --passes 400 --step-passes 400 --step-horizon 240 --quarters 264 --step-size 0.03 --tag _cut_main
$RUN --rate-lower 0 --foresight --passes 700 --quarters 264 --step-size 0.03 --tag _cut_foresight
$RUN --rate-lower 0 --passes 400 --step-passes 400 --step-horizon 240 --quarters 264 --step-size 0.03 --capitalist-mpc 0.25 --tag _cut_mpc25
$RUN --rate-lower 0 --passes 400 --step-passes 400 --step-horizon 240 --quarters 264 --step-size 0.03 --capitalist-mpc 0.75 --tag _cut_mpc75

# ---- Section 7.2: the licence-rent transfer (one half), and the whole rent in its footnote
$RUN --passes 300 --step-passes 200 --licence-transfer 0.5 --tag _transfer50
$RUN --passes 300 --step-passes 200 --licence-transfer 1.0 --tag _transfer100

# ---- Footnote of Section 6.3: the limit case with public purchases
# The sixth and the last yearly step settle only with a smaller solver step; a step that does not
# settle is solved again with half the step (--retries, on by default).
$RUN --public-purchases --passes 400 --step-passes 150 --step-size 0.02 --memory 12 --tag _purchases

# ---- Footnote of Section 5.2: elasticity of one between tasks; AI 30 percent cheaper than knowledge labor
$RUN --passes 300 --step-passes 200 --eta 1.0 --tag _eta10
$RUN --passes 300 --step-passes 200 --ai-discount 0.30 --tag _ai30

# ---- Footnotes of Sections 5.3 and 8.2: other assumptions on the entry into manual work
# no applications of employed knowledge workers for manual jobs
$RUN --passes 300 --step-passes 200 --offer-knowledge 0 --tag _noapp
$RUN --passes 300 --step-passes 200 --offer-knowledge 0 --open-access --tag _noapp_access
$RUN --passes 300 --step-passes 200 --offer-knowledge 0 --open-access --no-capability --tag _noapp_access_noai
# earnings in manual work of 80 and of 100 percent of home earnings
$RUN --passes 300 --step-passes 200 --rho-knowledge 0.8 --tag _rho80
$RUN --passes 300 --step-passes 200 --rho-knowledge 0.8 --open-access --tag _rho80_access
$RUN --passes 300 --step-passes 200 --rho-knowledge 0.8 --open-access --no-capability --tag _rho80_access_noai
$RUN --passes 300 --step-passes 200 --rho-knowledge 1.0 --tag _rho100
$RUN --passes 300 --step-passes 200 --rho-knowledge 1.0 --open-access --tag _rho100_access
$RUN --passes 300 --step-passes 200 --rho-knowledge 1.0 --open-access --no-capability --tag _rho100_access_noai
# a quarter of the knowledge-home entrants qualified for manual work on entry
$RUN --passes 300 --step-passes 200 --knowledge-access 0.25 --tag _acc25

# ---- Tables, figures and numbers
$PY -m capability_model.report
