"""Command line for the capability model.

    python -m capability_model.run reference     reference equilibrium and its fit
    python -m capability_model.run scenarios     stationary equilibria at higher capability
    python -m capability_model.run transition    transition paths for the main case

Results are written to results/. Reference equilibria are computed once and kept in
results/stationary_cache.
"""
import argparse, json, pickle, time
from dataclasses import replace
import numpy as np
from . import parameters as par, stationary as st, transition as tr
from .households import N, M, PRIVATE, emp, unemp

OUT = par.ROOT / 'results'
FRONTIERS = [0.19, 0.35, 0.50, 0.65]
CASES = [(0.5, 0.2), (0.5, 0.5), (0.8, 0.2), (0.8, 0.5)]      # (elasticity, AI cost below labour)
MAIN = (0.5, 0.5)                                              # main scenario: AI 50% cheaper (author, 14 November 2026)
OLD_MODEL = {
    'private employment, million': 118.18, 'knowledge employment share, %': 60.47,
    'private unemployment rate, %': 2.038, 'private residents, million': 120.64,
    'inactive, million': 52.11, 'retired, million': 47.68, 'other employed, million': 44.51,
    'zero liquid assets, all residents, %': 4.47, 'liquid assets, model units': 1.7914,
    'N to M, unemployed origin, % per quarter': 11.38, 'N to M, employed origin, % per quarter': 0.284,
    'M to N, unemployed origin, % per quarter': 9.40, 'M to N, employed origin, % per quarter': 0.984,
}


def fit(const, res):
    ag = res['households']['ag']; sc = const['persons_per_unit'] / 1e6
    E, U, O = ag['employment'], ag['unemployment'], ag['outside']
    fl, op = ag['status_flows'], ag['status_open']
    return {
        'private employment, million': E.sum() * sc, 'knowledge employment share, %': 100 * E[N] / E.sum(),
        'private unemployment rate, %': 100 * U.sum() / (E.sum() + U.sum()),
        'private residents, million': (E.sum() + U.sum()) * sc,
        'inactive, million': O[0] * sc, 'retired, million': O[1] * sc, 'other employed, million': O[2] * sc,
        'zero liquid assets, all residents, %': 100 * ag['zero_assets'], 'liquid assets, model units': ag['assets_open'],
        'N to M, unemployed origin, % per quarter': 100 * fl[unemp(N), emp(M)] / op[unemp(N)],
        'N to M, employed origin, % per quarter': 100 * fl[emp(N), emp(M)] / op[emp(N)],
        'M to N, unemployed origin, % per quarter': 100 * fl[unemp(M), emp(N)] / op[unemp(M)],
        'M to N, employed origin, % per quarter': 100 * fl[emp(M), emp(N)] / op[emp(M)],
    }


def build(eta=0.5, ai_discount=0.2, unable_share=0.0, scarce_share=par.SCARCE_SHARE, payout_share=None, offer_knowledge=None,
          knowledge_access=None, rho_knowledge=None, **policy):
    P, const = par.load(unable_share=unable_share)
    if knowledge_access is not None:
        # a check (17 November 2026): another share of knowledge-home entrants who are qualified for manual work
        # without instruction (par.KNOWLEDGE_ACCESS, 0.5, in the calibration)
        P.entry_access = np.where(P.home == N, knowledge_access, P.entry_access)
    if rho_knowledge is not None:
        # a check (17 November 2026): another ratio of the earnings of knowledge-home residents in manual work to their
        # home earnings at reference wages (0.9005 for almost all of them in the calibration); their efficiency units in
        # manual work are scaled in the same proportion for every knowledge-home type
        P.eff = P.eff.copy(); P.eff[P.home == N, M] *= rho_knowledge / RHO_KNOWLEDGE
    if offer_knowledge is not None:
        # a diagnostic (17 November 2026): the intensity with which employed knowledge workers who are qualified for
        # manual work apply for manual jobs (0.92 in the calibration; they accept about 0.01 percent of the offers)
        P.offer_intensity = P.offer_intensity.copy(); P.offer_intensity[N] = offer_knowledge
    if payout_share is not None:
        # robustness: residents hold this share of the claims on business payouts. Every resident receives the same
        # additional claim, on top of the claims of the residents who receive dividends in the data.
        P.claim = P.claim + (payout_share - float(P.mass @ P.claim)) / float(P.mass.sum())
    for key, value in policy.items():
        setattr(P, key, value)
    eco, ref = st.reference(P, const['wage_ref'], const['wage_share'], const['beta_C'],
                            par.technology_case(const, eta, ai_discount, scarce_share=scarce_share),
                            gdp_target=const['gdp_ref'])
    return eco, ref, const


def plain(x):
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def reference(args):
    t = time.time()
    eco, ref, const = build(*MAIN)
    ag, acc = ref['households']['ag'], ref['accounts']
    rows = fit(const, ref)
    print(f"reference solved in {time.time() - t:.0f}s; value gap {ref['households']['sol']['gap']:.1e}; "
          f"resource gap {acc['resource_gap']:.1e}; household budget gap {ag['budget_gap']:.1e}")
    print(f"{'margin':45s} {'new model':>10s} {'old model':>10s} {'data':>10s}")
    for k, v in rows.items():
        print(f"{k:45s} {v:10.3f} {OLD_MODEL[k]:10.3f} {par.TARGETS[k]:10.3f}")
    people = const['persons_per_unit'] / 1e3
    print(f"output {acc['output']:.4f}; payouts {acc['payouts']:.4f}; capitalist consumption "
          f"{acc['capitalist_consumption']:.4f}; recruiting cost per vacancy {eco.vacancy_cost.round(4)}")
    print(f"markup over cost {1 / eco.tech.marginal_cost:.3f}; levy {acc['levy']:.4f}; household consumption "
          f"{acc['household_consumption']:.4f}; investment {acc['investment']:.4f}")
    print(f"training: {ag['participants'] * people:.1f} thousand participants and "
          f"{ag['qualified'] * people:.1f} thousand qualifications per quarter")
    print(f"qualified for both occupations, by home: {ag['access_share'].round(4)}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'reference.json').write_text(json.dumps(plain(dict(
        scope='Reference stationary equilibrium with inherited, not recalibrated, parameters.',
        fit=dict(new_model=rows, old_model=OLD_MODEL, data=par.TARGETS),
        summary=st.summary(eco, ref), types=const['types'], vacancy_cost=eco.vacancy_cost,
        technology=eco.tech.__dict__)), indent=2) + '\n', encoding='utf-8')
    print(f"written: {(OUT / 'reference.json').relative_to(par.ROOT)}")


def scenarios(args):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for eta, discount in CASES:
        for regime, policy in (('closed', dict(training_open=False)), ('self-financed', dict())):
            t = time.time()
            eco, ref, const = build(eta, discount, **policy)
            base = st.summary(eco, ref)
            prev, level = ref, eco.tech.frontier
            for frontier in FRONTIERS:
                e = st.with_frontier(eco, frontier)
                res = st.solve_path(eco, prev, level, frontier)
                if not res['converged']:
                    print(f"eta {eta} AI -{round(100 * discount)}% {regime:13s} frontier {round(100 * frontier)}%: "
                          f"NOT SOLVED (stopped at {100 * res['reached']:.1f}%, residual {res['max_residual']:.1e})", flush=True)
                    rows.append(dict(elasticity=eta, ai_cost_below_labour_pct=round(100 * discount), training=regime,
                                     frontier_pct=round(100 * frontier), converged=False,
                                     reached_pct=100 * res['reached'], max_residual=res['max_residual']))
                    break
                level = frontier
                s = st.summary(e, res, ref)
                kh, kh0 = s['knowledge_home'], base['knowledge_home']
                rows.append(dict(
                    elasticity=eta, ai_cost_below_labour_pct=round(100 * discount), training=regime,
                    frontier_pct=round(100 * frontier), converged=res['converged'],
                    max_residual=res['max_residual'], resource_gap=res['accounts']['resource_gap'],
                    automated_share_pct=100 * s['automated_share'], **{k + '_pct': v for k, v in s['change'].items()},
                    kh_knowledge_job_pct=100 * kh['knowledge_job'], kh_manual_job_pct=100 * kh['manual_job'],
                    kh_unemployed_pct=100 * kh['unemployed'],
                    **{'kh_' + k + '_pct': 100 * v for k, v in kh['outside'].items()},
                    kh_private_job_ref_pct=100 * (kh0['knowledge_job'] + kh0['manual_job']),
                    kh_with_manual_access_pct=100 * kh['with_manual_access'],
                    participants_thousand=s['participants'] * const['persons_per_unit'] / 1e3,
                    qualified_thousand=s['qualified'] * const['persons_per_unit'] / 1e3))
                prev = res
                r = rows[-1]
                print(f"eta {eta} AI -{round(100 * discount)}% {regime:13s} frontier {r['frontier_pct']:2d}%: "
                      f"knowledge jobs {r['knowledge_employment_pct']:+6.1f}%  manual jobs {r['manual_employment_pct']:+6.1f}%  "
                      f"knowledge wage {r['knowledge_wage_pct']:+6.1f}%  manual wage {r['manual_wage_pct']:+6.1f}%  "
                      f"output {r['output_pct']:+6.1f}%  residual {r['max_residual']:.0e}", flush=True)
            print(f"  case done in {time.time() - t:.0f}s", flush=True)
            (OUT / 'scenarios.json').write_text(json.dumps(plain(dict(
                scope='Stationary equilibria of the full model; inherited parameters; no transition path.',
                rows=rows)), indent=2) + '\n', encoding='utf-8')
    print(f"written: {(OUT / 'scenarios.json').relative_to(par.ROOT)}")


REGIMES = {'closed': dict(training_open=False), 'self-financed': dict()}
REPORT_QUARTERS = [0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 60, 80, 120, 159]


RHO_KNOWLEDGE = 0.9005230356716936                             # the calibrated ratio (paired_relative_earnings of the inherited parameters)


def stationary_case(eta, discount, regime, frontier=None, rent_share=None, payout_share=None, offer_knowledge=None,
                    knowledge_access=None, rho_knowledge=None):
    """Reference and, if `frontier` is given, the stationary equilibrium there; solved once and kept on disk."""
    cache = OUT / 'stationary_cache'; cache.mkdir(parents=True, exist_ok=True)
    share = par.SCARCE_SHARE if rent_share is None else rent_share      # licence-rent share of the price of AI capital
    stem = (f'eta{eta}_ai{round(100 * discount)}_{regime}_iota{par.ENTRY_ELASTICITY}_{par.AI_SUPPLY}{share}'
            f'_access{par.KNOWLEDGE_ACCESS}' + ('' if payout_share is None else f'_own{payout_share}')
            + ('' if offer_knowledge is None else f'_offerK{offer_knowledge}')
            + ('' if knowledge_access is None else f'_accessK{knowledge_access}')
            + ('' if rho_knowledge is None else f'_rhoK{rho_knowledge}'))
    first = cache / f'{stem}_f05.pkl'
    if first.exists():
        d = pickle.load(open(first, 'rb')); eco, ref, const = d['eco'], d['res'], d['const']
    else:
        eco, ref, const = build(eta, discount, scarce_share=share, payout_share=payout_share, offer_knowledge=offer_knowledge,
                                knowledge_access=knowledge_access, rho_knowledge=rho_knowledge, **REGIMES[regime])
        pickle.dump(dict(eco=eco, res=ref, const=const), open(first, 'wb'))
    if frontier is None:
        return eco, ref, const, None
    last = cache / f'{stem}_f{round(100 * frontier):02d}.pkl'
    if last.exists():
        end = pickle.load(open(last, 'rb'))['res']
    else:
        solved = sorted((int(f.stem[-2:]) / 100, f) for f in cache.glob(f'{stem}_f*.pkl') if int(f.stem[-2:]) / 100 < frontier)
        level, start = solved[-1][0], pickle.load(open(solved[-1][1], 'rb'))['res']
        end = st.solve_path(eco, start, level, frontier)
        if end['converged']:
            pickle.dump(dict(eco=st.with_frontier(eco, frontier), res=end), open(last, 'wb'))
    return eco, ref, const, end


def cohort_shares(P, cohort):
    """Shares of the surviving initial knowledge cohort: knowledge job, manual job, unemployed, outside."""
    total = cohort.sum(1)
    return np.column_stack([cohort[:, emp(N)], cohort[:, emp(M)], cohort[:, unemp(N)] + cohort[:, unemp(M)],
                            cohort[:, PRIVATE:].sum(1)]) / total[:, None]


def transition(args):
    OUT.mkdir(parents=True, exist_ok=True)
    eta, discount = MAIN
    if args.eta is not None:                               # checks: another elasticity between the two kinds of tasks,
        eta = args.eta
    if args.ai_discount is not None:                       # another cost of AI relative to knowledge labour
        discount = args.ai_discount
    T, results = args.quarters, []
    for regime in args.regimes:
        for target in args.frontiers:
            t = time.time()
            eco, ref, const, end = stationary_case(eta, discount, regime, target if args.end == 'stationary' else None,
                                                   rent_share=args.rent_share, payout_share=args.payout_share,
                                                   offer_knowledge=args.offer_knowledge, knowledge_access=args.knowledge_access,
                                                   rho_knowledge=args.rho_knowledge)
            if end is not None and not end['converged']:
                print(f"{regime} {round(100 * target)}%: the final stationary equilibrium was not solved "
                      f"(stopped at {100 * end['reached']:.1f}%); no path computed", flush=True)
                continue
            frontier = tr.frontier_path(eco.tech.frontier, target, par.PATH_QUARTERS, T)
            eco = replace(eco, tech=replace(eco.tech, smooth_C=par.INVESTMENT_SMOOTHING, smooth_A=par.INVESTMENT_SMOOTHING))
            saved = OUT / 'stationary_cache' / f'path_{regime}_f{round(100 * target)}.pkl'
            guess = pickle.load(open(saved, 'rb')) if args.warm and saved.exists() else None
            if guess is not None and len(guess['D']) != T:
                guess = None
            mpc = par.CAPITALIST_MPC if args.capitalist_mpc is None else args.capitalist_mpc
            # the return on claims is fixed at its reference value; with --rate-lower it follows the rule of the variant
            response = 0.0 if args.rate_lower is None else par.RATE_RESPONSE
            options = dict(returns=args.returns, mpc=mpc, invest_weight=par.INVEST_CAPACITY_WEIGHT,
                           rate_response=response,
                           rate_lower=None if args.rate_lower is None else (1 + args.rate_lower / 100) ** 0.25)
            if args.step_size is not None:
                options['step'] = args.step_size
            if args.anchor_speed is not None:
                options['anchor_speed'] = args.anchor_speed
            if args.income_speed is not None:               # a test: capitalists respond to a smoothed income
                options['anchor_speed'], options['delayed'] = args.income_speed, True
            if args.licence_transfer is not None:           # policy experiment: a share of the rise in the licence rent goes to residents
                options['transfer_rate'] = args.licence_transfer
            if args.public_purchases:                       # policy experiment: public purchases keep capacity in use
                options['purchases'] = True
            if args.memory is not None:
                options['memory'] = args.memory
            if args.plain_passes is not None:
                options['burn'] = args.plain_passes
            if args.tolerance is not None:
                options['tol'] = args.tolerance
            wage_speed = par.WAGE_SPEED if args.wage_speed is None else args.wage_speed
            eco_ref, start = eco, None
            if args.open_access:                             # easier access: all knowledge-home residents and entrants qualified
                eco = replace(eco, P=replace(eco.P, entry_access=np.where(eco.P.home == N, 1.0, eco.P.entry_access)))
                start = tr.open_access(eco.P, tr.start_state(eco, ref))
            if par.EXPECTATIONS == 'surprise' and not args.foresight:
                n = par.SURPRISE_QUARTERS
                levels = [float(frontier[i + n // 2]) for i in range(0, par.PATH_QUARTERS, n)]     # level in the middle of each step
                final = target
                if args.no_capability:                       # control: capability stays at its initial level
                    levels, final = [], float(eco.tech.frontier)
                out = tr.solve_surprises(eco, ref, levels, n, final, T, wage_speed, max_iter=min(args.passes, args.step_passes),
                                         slump_start=args.slump_start, step_horizon=args.step_horizon, start=start,
                                         checkpoint=args.checkpoint, retries=args.retries,
                                         final_iter=args.passes, verbose=args.verbose, **options)
                out['guess'] = None
            else:
                out = tr.solve(eco, ref, end, frontier, wage_speed, max_iter=args.passes, verbose=args.verbose,
                               guess=guess, **options)
            p, sc = out['paths'], const['persons_per_unit'] / 1e6
            base = tr.reference_cohort(eco_ref, ref, T)
            sh, sh0 = cohort_shares(eco.P, p['cohort']), cohort_shares(eco.P, base)
            q = [s for s in REPORT_QUARTERS if s < T]
            w0 = eco.wage_share * ref['lam']
            kh = p['status_by_home'][:, N] / p['status_by_home'][:, N].sum(1, keepdims=True)
            row = dict(
                elasticity=eta, ai_cost_below_labour_pct=round(100 * discount), training=regime,
                final_frontier_pct=round(100 * target), path_quarters=par.PATH_QUARTERS, horizon=T,
                wage_speed=wage_speed, entry_elasticity=par.ENTRY_ELASTICITY, returns=args.returns, end=args.end,
                expectations='foresight' if args.foresight else par.EXPECTATIONS, steps=out.get('steps'),
                ai_rent_share=par.SCARCE_SHARE if args.rent_share is None else args.rent_share, ai_supply=par.AI_SUPPLY, capitalist_mpc=mpc,
                knowledge_access=par.KNOWLEDGE_ACCESS, invest_capacity_weight=par.INVEST_CAPACITY_WEIGHT,
                investment_smoothing=par.INVESTMENT_SMOOTHING, rate_response=response,
                rate_lower_annual_pct=args.rate_lower, anchor_speed=args.anchor_speed, income_speed=args.income_speed,
                open_access=args.open_access, no_capability=args.no_capability,
                licence_transfer=args.licence_transfer, public_purchases=args.public_purchases,
                utilisation_pct=100 * p['utilisation'][q], capacity_pct=100 * (p['capacity'][q] / ref['accounts']['output'] - 1),
                capitalist_consumption_pct=100 * (p['capitalist_consumption'][q] / ref['accounts']['capitalist_consumption'] - 1),
                capitalist_saving_pct_of_output=100 * p['capitalist_saving'][q] / p['output'][q],
                wage_bill_pct=100 * (p['payroll'][q] / ref['households']['ag']['payroll'] - 1),
                automated_share_pct=100 * p['automated'][q], ai_price=p['ai_price'][q],
                annual_return_pct=100 * (p['R'][q] ** 4 - 1),
                unemployment_rate_pct=100 * p['unemployment'][q].sum(1) / (p['unemployment'][q].sum(1) + p['employment'][q].sum(1)),
                knowledge_home_qualified_pct=100 * p['knowledge_home_access'][q],
                passes=out['passes'], max_gap=out['max_gap'], converged=out['converged'], resource_gap=out['resource_gap'],
                quarters=q, frontier_pct=(100 * p['frontier'][q]), automated_pct=(100 * p['automated'][q]),
                knowledge_jobs_million=p['employment'][q, N] * sc, manual_jobs_million=p['employment'][q, M] * sc,
                unemployed_million=p['unemployment'][q].sum(1) * sc, outside_million=p['outside'][q].sum(1) * sc,
                knowledge_wage_pct=100 * (p['wage'][q, N] / w0[N] - 1), manual_wage_pct=100 * (p['wage'][q, M] / w0[M] - 1),
                knowledge_finding=p['finding'][q, N], manual_finding=p['finding'][q, M],
                output_pct=100 * (p['output'][q] / ref['accounts']['output'] - 1),
                household_consumption_pct=100 * (p['household_consumption'][q] / ref['accounts']['household_consumption'] - 1),
                layoffs_knowledge_million=p['layoffs'][q, N] * sc, layoffs_manual_million=p['layoffs'][q, M] * sc,
                participants_thousand=p['participants'][q] * sc * 1e3, qualified_thousand=p['qualified'][q] * sc * 1e3,
                cohort_groups=['knowledge job', 'manual job', 'unemployed', 'outside the private workforce'],
                cohort_share_pct=100 * sh[q], cohort_share_reference_pct=100 * sh0[q],
                knowledge_home_private_job_pct=100 * (kh[q, emp(N)] + kh[q, emp(M)]),
                knowledge_home_private_job_reference_pct=100 * (ref['households']['mu_income'][eco.P.home == N][:, :, [emp(N), emp(M)]].sum()
                                                               / ref['households']['mu_income'][eco.P.home == N].sum()))
            results.append(row)
            base_groups = tr.reference_groups(eco_ref, ref, T)       # the cohort by qualification and savings at the start
            np.savez_compressed(OUT / f'transition_{regime}_f{round(100 * target)}{args.tag}.npz',
                                cohort_reference=base, cohort_groups_reference=base_groups['groups'],
                                cohort_groups_consumption_reference=base_groups['groups_consumption'],
                                cohort_groups_assets_reference=base_groups['groups_assets'],
                                # values of residents at the first date, by state: as they turn out and as expected then
                                **({} if out.get('welfare_start') is None else
                                   dict(welfare_start=out['welfare_start'], welfare_expected=out['welfare_expected'])),
                                **{k: v for k, v in p.items()})
            if out.get('welfare_steps') is not None:        # further welfare numbers (welfare.py): other discounting, later dates
                from . import welfare
                welfare.write(eco, eco_ref, ref, np.load(OUT / f'transition_{regime}_f{round(100 * target)}{args.tag}.npz'),
                              out['welfare_steps'], args.tag)
            if out['guess'] is not None and not args.tag:
                pickle.dump(out['guess'], open(OUT / 'stationary_cache' / f'path_{regime}_f{round(100 * target)}.pkl', 'wb'))
            print(f"{regime}, frontier to {round(100 * target)}%: {out['passes']} passes, largest gap {out['max_gap']:.1e}, "
                  f"{time.time() - t:.0f}s", flush=True)
            print('  quarter  AI tasks  sold/capacity  output  household C  knowledge jobs  manual jobs  unemployed  outside  '
                  'knowledge wage  manual wage  cohort without a private job (reference)')
            for i, s in enumerate(q):
                print(f"  {s:7d}  {row['automated_share_pct'][i]:7.1f}%  {row['utilisation_pct'][i]:12.1f}%  {row['output_pct'][i]:+5.1f}%  "
                      f"{row['household_consumption_pct'][i]:+10.1f}%  {row['knowledge_jobs_million'][i]:14.1f}  "
                      f"{row['manual_jobs_million'][i]:11.1f}  {row['unemployed_million'][i]:10.2f}  "
                      f"{row['outside_million'][i]:7.1f}  {row['knowledge_wage_pct'][i]:+13.1f}%  {row['manual_wage_pct'][i]:+10.1f}%   "
                      f"{100 * (sh[s, 2] + sh[s, 3]):5.1f}% ({100 * (sh0[s, 2] + sh0[s, 3]):.1f}%)", flush=True)
            (OUT / f'transition{args.tag}.json').write_text(json.dumps(plain(dict(
                scope='Transition paths of the full model; inherited parameters; return on claims held fixed '
                      'unless returns is "euler".', rows=results)), indent=2) + '\n', encoding='utf-8')
    print(f"written: {(OUT / f'transition{args.tag}.json').relative_to(par.ROOT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['reference', 'scenarios', 'transition'])
    # the main scenario: capability rises to 35% of knowledge tasks and training is self-financed
    parser.add_argument('--frontiers', type=float, nargs='+', default=[0.35], help='final frontier (transition)')
    parser.add_argument('--regimes', nargs='+', default=['self-financed'], choices=list(REGIMES))
    parser.add_argument('--quarters', type=int, default=160, help='length of the solved path')
    parser.add_argument('--end', default='free', choices=['free', 'stationary'],
                        help='free: prices stay at their last values after the path; stationary: solve the final equilibrium')
    parser.add_argument('--passes', type=int, default=150, help='largest number of passes over the path')
    parser.add_argument('--returns', default='policy', choices=['policy', 'fixed', 'euler'],
                        help='policy: the return is fixed (or follows the rule of --rate-lower); solved with the capital paths')
    parser.add_argument('--rate-lower', type=float, default=None,
                        help='variant: the return follows capacity use (parameters.RATE_RESPONSE) and cannot fall below '
                             'this annual return in percent (default: the return is fixed at its reference value)')
    parser.add_argument('--step-passes', type=int, default=80, help='largest number of passes for each yearly step')
    parser.add_argument('--anchor-speed', type=float, default=None,
                        help='robustness: share of the gap to their current income by which the anchor of '
                             'capitalists\' consumption moves each quarter (default: the anchor is fixed)')
    parser.add_argument('--income-speed', type=float, default=None,
                        help='a test: capitalists consume out of a smoothed income that moves this share of the way toward '
                             'their current income each quarter (default: they respond to current income)')
    parser.add_argument('--wage-speed', type=float, default=None,
                        help='robustness: share of the gap to the flexible wage that posted wages close each quarter '
                             '(default: parameters)')
    parser.add_argument('--payout-share', type=float, default=None,
                        help='robustness: share of the claims on business payouts held by residents, the additional '
                             'claims spread equally over all residents (default: the claims in the data, about 1 percent)')
    parser.add_argument('--open-access', action='store_true',
                        help='experiment: from the first date every knowledge-home resident and entrant is qualified '
                             'for manual work (default: qualification as in the reference)')
    parser.add_argument('--licence-transfer', type=float, default=None,
                        help='policy experiment: share of the rise of the licence rent above its reference level that is paid '
                             'to residents, the same amount to each, financed by the levy on business cash (default: none)')
    parser.add_argument('--public-purchases', action='store_true',
                        help='policy experiment: the government buys final goods whenever spending would fall short of normal '
                             'capacity, financed by the levy on business cash (default: no purchases)')
    parser.add_argument('--offer-knowledge', type=float, default=None,
                        help='a diagnostic: intensity with which employed knowledge workers apply for manual jobs (calibration: 0.92)')
    parser.add_argument('--knowledge-access', type=float, default=None,
                        help='a check: share of knowledge-home entrants qualified for manual work without instruction (calibration: 0.5)')
    parser.add_argument('--rho-knowledge', type=float, default=None,
                        help='a check: earnings of knowledge-home residents in manual work relative to home earnings (calibration: 0.90)')
    parser.add_argument('--no-capability', action='store_true',
                        help='control: capability stays at its initial level; one path with correct expectations')
    parser.add_argument('--rent-share', type=float, default=None,
                        help='robustness: licence-rent share of the price of AI capital (default: parameters)')
    parser.add_argument('--capitalist-mpc', type=float, default=None,
                        help='robustness: share of an increase in their income that capitalists consume (default: parameters)')
    parser.add_argument('--step-size', type=float, default=None, help='size of the solver step over the paths (default 0.05)')
    parser.add_argument('--memory', type=int, default=None,
                        help='passes used by the acceleration of the solver (default 6; more when the paths settle slowly)')
    parser.add_argument('--plain-passes', type=int, default=None,
                        help='plain passes before the solver extrapolates (default 12); a large number switches extrapolation off')
    parser.add_argument('--tolerance', type=float, default=None, help='largest gap accepted on the paths (default 0.001)')
    parser.add_argument('--step-horizon', type=int, default=100,
                        help='quarters that each yearly step looks ahead (longer when adjustment is slow)')
    parser.add_argument('--slump-start', type=float, default=None,
                        help='test only: start every step from this rate of capacity use and the lowest return')
    parser.add_argument('--foresight', action='store_true',
                        help='a test: the whole path of capability is foreseen from the first date (no yearly surprises)')
    parser.add_argument('--eta', type=float, default=None,
                        help='check: elasticity of substitution between knowledge and manual tasks (default: 0.5); the reference '
                             'is computed again with it')
    parser.add_argument('--ai-discount', type=float, default=None,
                        help='check: share by which AI costs less than knowledge labour on feasible tasks at reference prices '
                             '(default: 0.5); the reference is computed again with it')
    parser.add_argument('--checkpoint', default=None,
                        help='file in which the state is saved after every yearly step; a run that finds it continues from there')
    parser.add_argument('--retries', type=int, default=2,
                        help='times a yearly step that does not reach the tolerance is solved again with half the solver step')
    parser.add_argument('--tag', default='', help='suffix for the output files, so that a test does not replace the main results')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--warm', action='store_true', help='start from the path saved by the previous run')
    args = parser.parse_args()
    {'reference': reference, 'scenarios': scenarios, 'transition': transition}[args.command](args)


if __name__ == '__main__':
    main()
