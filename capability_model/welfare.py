"""Further welfare numbers for a solved path (Appendix B.3 of the paper, "Welfare").

    python -m capability_model.welfare --tag _transfer50 --steps FILE [--open-access] [--payout-share 0.30]

A run of `capability_model.run transition` writes the values of residents at the initial date as they turn
out, at the residents' own discount factors (welfare_start in its .npz file). This module adds, in the file
results/welfare{tag}.npz:

  start_social   the same values when every date is discounted at the return on claims of the reference instead of
                 the residents' own discount factors (their choices are unchanged; only the weights on dates differ);
  later_own,     the values, by state, of a resident who is in that state at the start of a later quarter (ten,
  later_social   twenty and forty years after the first rise in capability), at the two discount rates;
  entrants       the distribution over states of the residents who enter in a quarter of the reference.

After the last step of capability expectations are correct, so the values at later dates are those of the
household problem along the saved path. Before it, the policies that residents follow under the expectations of
each step are needed; `--steps` names the file in which the run saved them (run.py --checkpoint). No equilibrium
is solved here.
"""
import argparse
import json
import pickle
from dataclasses import replace
import numpy as np
from . import parameters as par, households as hh, run
from .households import N

OUT = par.ROOT / 'results'
LATER = (40, 80, 159)                                       # quarters: ten, twenty and forty years


def evaluator(P, beta):
    """The same residents with the discount factor `beta` for every type (used to value paths, not to choose)."""
    return replace(P, beta=np.full(P.nh, float(beta)))


def stationary_value(P, Pe, sol, tol=1e-10, max_iter=50000):
    """Values of the stationary policies `sol` when dates are discounted with the factors of `Pe`."""
    cost = hh.apply_transitions(P, sol['T'], hh.policy_value(P, sol['policy'], sol['V'])) - sol['V']     # expected choice costs
    W = sol['V'].copy()
    for _ in range(max_iter):
        new = hh.apply_transitions(P, sol['T'], hh.policy_value(Pe, sol['policy'], W)) - cost
        gap = np.max(np.abs(np.where(np.isfinite(new - W), new - W, 0.0))); W = new
        if gap < tol:
            break
    return W


def reference_parts(P, sol, beta=None):
    """The consumption part of the reference values and the rest, at the residents' own discount factors or at `beta`."""
    if beta is None:
        A = hh.consumption_part(P, sol)
        return A, sol['V'] - A
    Pe = evaluator(P, beta)
    A = hh.consumption_part(Pe, sol)
    return A, stationary_value(P, Pe, sol) - A


def later(eco, ref, p, start, beta, dates=LATER):
    """The household problem along the saved path from date `start` on, when expectations are correct.

    Returns the values at the start of quarter `start` and of the quarters in `dates`, at the residents' own
    discount factors (the values of the problem) and at the factor `beta` (the same policies, other weights on dates).
    """
    P, Pe = eco.P, evaluator(eco.P, beta)
    T = len(p['output']); mass = float(ref['households']['mu'].sum())
    S = p['transfer'] if 'transfer' in p.files else np.zeros(T)
    R_end = float(p['R'][T - 1])
    last = hh.Prices(R=R_end, D=p['D'][T - 1], wage=eco.wage_share * p['lam'][T - 1], finding=p['finding'][T - 1], transfer=S[T - 1] / mass)
    sol = hh.solve_values(P, last, ref['households']['sol']['V'])          # prices stay at their last values after the path
    V, W = sol['V'], stationary_value(P, Pe, sol)
    cont = np.maximum(hh.job_values(P, last, sol, p['lam'][T - 1], R_end), 0.0) * hh.keep_probability(P, sol['choice'])
    own, social = {}, {}
    for t in range(T - 1, start - 1, -1):
        pr = hh.Prices(R=p['R'][t], D=p['D'][t], wage=p['wage'][t], finding=p['finding'][t], transfer=S[t] / mass)
        Vbar, policy = hh.income_stage(P, pr, V)
        J = hh.job_value_step(P, p['wage'][t], p['lam'][t], policy, p['R'][t + 1] if t + 1 < T else R_end, cont)
        pr.hire_ok = (J >= 0).astype(float); pr.layoff = (J < 0).astype(float)
        V, Tm, choice = hh.start_stage(P, pr, Vbar)
        cont = np.maximum(J, 0.0) * hh.keep_probability(P, choice)
        W = hh.apply_transitions(P, Tm, hh.policy_value(Pe, policy, W)) - (hh.apply_transitions(P, Tm, Vbar) - V)
        if t in dates or t == start:
            own[t], social[t] = V.copy(), W.copy()
    return own, social


def realised(P, steps, value, Pe=None):
    """Values at the initial date as they turn out: backward through the retained dates of the steps before the last,
    with the policies, status probabilities and choice costs of each step and the values that actually follow."""
    for step in reversed(steps):
        for policy, Tm, cost in reversed(step['dates']):
            value = hh.apply_transitions(P, Tm, hh.policy_value(P if Pe is None else Pe, policy, value)) - cost
    return value


def entrants(eco, ref):
    """Distribution over start-of-quarter states of the residents who enter in a quarter of the reference."""
    P, sol = eco.P, ref['households']['sol']
    mui = hh.income_distribution(P, ref['households']['mu'], sol['T'])
    return hh.advance(P, sol['policy'], mui, entrants=True) - hh.advance(P, sol['policy'], mui, entrants=False)


def write(eco, eco_ref, ref, p, steps, tag, beta=None):
    """Compute the further welfare numbers of one path and save them. `steps` are the welfare pieces of the steps
    before the last one (empty when capability does not rise in steps); `eco` is the economy of the path and
    `eco_ref` that of the reference (they differ when access is made easier)."""
    beta = 1 / eco_ref.R if beta is None else beta
    start = sum(len(s['dates']) for s in steps)
    dates = [t for t in LATER if start <= t < len(p['output'])]
    own, social = later(eco, ref, p, start, beta, dates)
    check = realised(eco.P, steps, own[start])              # must reproduce what the run saved
    saved = p['welfare_start'] if 'welfare_start' in p.files else None
    on = ref['households']['mu'] > 0
    gap = float(np.max(np.abs(check - saved)[on])) if saved is not None else float('nan')
    np.savez_compressed(OUT / f'welfare{tag}.npz', start_social=realised(eco.P, steps, social[start], evaluator(eco.P, beta)),
                        start_own=check, later_own=np.array([own[t] for t in dates]), later_social=np.array([social[t] for t in dates]),
                        dates=np.array(dates), entrants=entrants(eco_ref, ref), beta=beta, check=gap)
    return gap


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tag', default='', help='suffix of the result files of the run')
    parser.add_argument('--steps', required=True, help='file written by run.py --checkpoint for the same run')
    parser.add_argument('--regime', default='self-financed')
    parser.add_argument('--frontier', type=float, default=0.35)
    parser.add_argument('--open-access', action='store_true', help='the run was made with --open-access')
    parser.add_argument('--payout-share', type=float, default=None, help='the run was made with this --payout-share')
    parser.add_argument('--rent-share', type=float, default=None, help='the run was made with this --rent-share')
    args = parser.parse_args()
    eco_ref, ref, const, _ = run.stationary_case(*run.MAIN, args.regime, None, rent_share=args.rent_share, payout_share=args.payout_share)
    eco = eco_ref
    if args.open_access:
        eco = replace(eco, P=replace(eco.P, entry_access=np.where(eco.P.home == N, 1.0, eco.P.entry_access)))
    p = np.load(OUT / f'transition_{args.regime}_f{round(100 * args.frontier)}{args.tag}.npz')
    saved = pickle.load(open(args.steps, 'rb'))
    steps = saved[6]
    row = json.load(open(OUT / f'transition{args.tag}.json'))['rows'][0]
    assert steps is not None and len(steps) == len(row['steps']) - 1, 'the file does not hold the steps before the last one'
    gap = write(eco, eco_ref, ref, p, steps, args.tag)
    print(f"written: welfare{args.tag}.npz; the values at the initial date are reproduced to {gap:.1e}")


if __name__ == '__main__':
    main()
