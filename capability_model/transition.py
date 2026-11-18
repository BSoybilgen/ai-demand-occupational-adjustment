"""Transition paths (Section 4.7 of the paper, "Reference and experiments").

The economy starts in the reference stationary equilibrium. A path for the
capability frontier is announced at date 0 and foreseen. Dates 0..T-1 are solved.
The end of the path is either a stationary equilibrium supplied by the caller
(`end`), or, with end=None, the assumption that prices, job finding and payouts
stay at their last values after date T-1. The second form needs no stationary
equilibrium of the final frontier; it is accurate when the path has settled.

Unknown paths: the two service prices, the two job-finding probabilities, payouts
and the return on claims. One evaluation runs the households backward (values,
job values, hiring and layoff decisions) and forward (distribution), solves the
capital paths given labour, and returns the paths implied by production, free
entry, the accounts and the capitalists' saving condition. `solve` iterates on
the difference with damping.

Posted wages follow the gradual rule w_t = w_{t-1}^(1-phi) (share*lambda_t)^phi.

Demand (argument `mpc`). With mpc=None output is what firms can produce and
capitalists consume what is left. With a number, capitalists consume their
reference income plus the share `mpc` of any income above it. The return on claims
is fixed, so it cannot fall to turn the remaining saving into investment. If
Output then equals spending: it moves until capitalists' saving equals net
investment plus household dissaving. u_t is output relative to normal capacity and
is an additional unknown path. Employment accounts receive u_t times the service
price, and capital earns u_t times its rental, so unused capacity lowers hiring,
pay and investment. `invest_weight` in [0, 1] sets how fully investors take unused
capacity into account (capital earns u_t**weight times its rental; 0 = not at all).

Return on claims (argument `returns`). 'fixed': constant. 'policy' (used with mpc):
the return paid at t+1 is max(lower, floor * u_t**response), solved together with
capacity use and the capital paths (production.capital_path). With response 0, the
main scenario, the return is fixed at `floor`, its reference value. With a positive
response and `lower` below the floor (a variant) it rises when spending exceeds
normal capacity and falls, down to `lower`, when capacity is unused. 'euler' (not validated): capitalists' saving
condition.
"""
import os
import pickle
import numpy as np
from . import households as hh
from . import production as prod

N, M = hh.N, hh.M


def frontier_path(start, target, quarters, T):
    """Linear rise from `start` at date 0 to `target` at date `quarters`; length T+1."""
    return start + (target - start) * np.minimum(np.arange(T + 1) / quarters, 1.0)


def posted_wages(eco, lam, phi, start):
    w = np.empty_like(lam); prev = np.asarray(start, float)
    for t in range(len(lam)):
        w[t] = prev ** (1 - phi) * (eco.wage_share * lam[t]) ** phi
        prev = w[t]
    return w


def initial_guess(ref, end, frontier, R):
    T = len(frontier) - 1
    end = ref if end is None else end
    span = frontier[-1] - frontier[0]
    s = (frontier[:T] - frontier[0]) / span if span > 0 else np.zeros(T)
    mix = lambda a, b: np.outer(1 - s, np.atleast_1d(a)) + np.outer(s, np.atleast_1d(b))
    return dict(lam=mix(ref['lam'], end['lam']), finding=mix(ref['finding'], end['finding']),
                D=mix(ref['D'], end['D'])[:, 0], R=np.full(T + 1, R), K=None)


def reference_income(eco, ref):
    """Capitalists' income at the reference: business cash after replacement investment,
    household payouts and interest on household claims. Equals their consumption there."""
    acc, ag = ref['accounts'], ref['households']['ag']
    charges = (ag['payroll'] + acc['recruit'] + ag['receipts'][0] + acc['levy'] + acc['investment'] + acc['installation']
               + ag['claims'] * ref['D'] + (eco.R - 1) * ag['assets_open'])
    return acc['output'] - charges


def start_state(eco, ref):
    """State at the start of a path that begins in the reference."""
    mu = ref['households']['mu']
    cohort = np.zeros_like(mu)
    cohort[:, :, [hh.emp(N), hh.unemp(N)]] = mu[:, :, [hh.emp(N), hh.unemp(N)]]
    K = np.array([ref['allocation'].KC, ref['allocation'].KA])
    return dict(mu=mu, cohort=cohort, groups=cohort_groups(eco.P, cohort), K=K, wage=eco.wage_share * ref['lam'],
                I=np.array([eco.tech.dep_C, eco.tech.dep_A]) * K, advances=0.0)


def open_access(P, state):
    """Easier access (experiment, off by default): at the start of the path every knowledge-home
    resident who is not qualified for manual work becomes qualified. Statuses, assets and
    types are unchanged, and the groups of the cohort keep their labels."""
    def move(m):
        m = m.copy()
        for h in np.flatnonzero(P.home == N):
            m[..., h, 1, :, :] += m[..., h, 0, :, :]; m[..., h, 0, :, :] = 0.0
        return m
    return dict(state, mu=move(state['mu']), cohort=move(state['cohort']), groups=move(state['groups']))


def cohort_groups(P, cohort):
    """The initial knowledge cohort split by qualification and by savings at the start (reporting only).

    Qualification has three classes: knowledge-home members who cannot hold a manual job,
    knowledge-home members who can, and manual-home members. Savings has three classes: the
    thirds of the cohort when its members are ordered by their assets (members at an asset
    node that lies on a boundary are divided in proportion). Returns the masses of the nine
    groups; class c of qualification and third j of savings are at index 3 * c + j.
    """
    kn = P.home == N
    cls = np.zeros((3,) + cohort.shape)
    cls[0][kn, 0] = cohort[kn, 0]; cls[1][kn, 1] = cohort[kn, 1]; cls[2][~kn] = cohort[~kn]
    cum = np.r_[0.0, np.cumsum(cohort.sum((0, 1, 2)))] / cohort.sum()
    width = np.diff(cum)
    share = np.zeros((3, P.na))
    for j in range(3):
        over = np.clip(np.minimum(cum[1:], (j + 1) / 3) - np.maximum(cum[:-1], j / 3), 0.0, None)
        share[j] = np.divide(over, width, out=np.zeros(P.na), where=width > 0)
    return np.array([cls[c] * share[j] for c in range(3) for j in range(3)])


def follow_groups(P, groups, sol, t, track):
    """Records the statuses, consumption and assets of each group at date t and returns the survivors."""
    after = np.zeros_like(groups)
    for i, start in enumerate(groups):
        gi = hh.income_distribution(P, start, sol['T'])
        track['groups'][t, i] = gi.sum((0, 3))
        track['groups_consumption'][t, i] = (sol['policy']['cons'] * gi).sum()
        track['groups_assets'][t, i] = start.sum((0, 1, 2)) @ P.agrid
        after[i] = hh.advance(P, sol['policy'], gi, entrants=False)
    return after


def evaluate(eco, ref, end, frontier, guess, phi, returns='fixed', detail=False, mpc=None, invest_weight=1.0,
             rate_response=0.125, state=None, keep=None, feedback=0.7, rate_lower=None, anchor_speed=None, delayed=False,
             transfer_rate=None, purchases=False):
    """One backward-forward pass. Returns implied paths, gaps and the allocation.

    `feedback` is a solver setting: the share of a change in the wage bill that the
    capital-and-demand block lets households spend while it solves (see
    production.capital_path). It affects speed, not the solution.

    `state` is where the path starts (the reference if omitted). With `keep` = n and
    detail=True, the state after n dates is returned under 'state', so that a new
    path can start there.

    `anchor_speed` (robustness, off by default): the income at which capitalists'
    consumption is anchored is no longer fixed at its reference value; each quarter it
    moves this share of the way toward their current income. The anchor path is solved
    with the capital paths and capacity use; the guess only supplies a starting path.

    `delayed` (a test, off by default; needs `anchor_speed`): the anchor stays at its reference value, and the path that
    follows capitalists' income at speed `anchor_speed` is the income to which their consumption responds, so that
    they consume the reference income plus the share mpc of the excess of this smoothed income over it.

    `transfer_rate` (policy experiment, off by default): this share of the rise of the licence rent above its
    reference level is paid to residents, the same amount to each, and financed by the levy on business cash. The
    path of the transfer is carried in the guess under 'S' (total amount) and must reproduce itself.

    `purchases` (policy experiment, off by default): the government buys final goods whenever spending would fall
    short of normal capacity, financed by the levy. The guess of capacity use then carries the solved unknown of
    production.capital_path (below one: one minus the share of capacity that the government buys, in logs).
    """
    state = start_state(eco, ref) if state is None else state
    P, tech = eco.P, eco.tech
    T = len(frontier) - 1
    base_ref = reference_income(eco, ref)
    anchor0 = state.get('anchor', base_ref)
    anchor = None
    if anchor_speed is not None:
        anchor = guess.get('anchor')
        anchor = np.full(T, anchor0) if anchor is None else np.asarray(anchor, float)
    lam, finding, D, R = guess['lam'], guess['finding'], guess['D'], guess['R']
    u = np.ones(T) if mpc is None or guess.get('u') is None else guess['u']
    wage = posted_wages(eco, lam, phi, state['wage'])
    mass = float(state['mu'].sum())
    S = np.zeros(T) if transfer_rate is None or guess.get('S') is None else np.asarray(guess['S'], float)   # transfer, total

    # ---- end of the path -------------------------------------------------------
    if end is None:                                          # prices stay at their last values
        last = hh.Prices(R=R[T], D=D[T - 1], wage=eco.wage_share * lam[T - 1], finding=finding[T - 1], transfer=S[T - 1] / mass)
        sol = hh.solve_values(P, last, guess.get('V_end', ref['households']['sol']['V']))
        V, J_end, choice_end = sol['V'], hh.job_values(P, last, sol, lam[T - 1], R[T]), sol['choice']
    else:
        V, J_end, choice_end = end['households']['sol']['V'], end['households']['J'], end['households']['sol']['choice']
    V_end = V

    # ---- backward: values, job values, hiring and layoffs --------------------
    cont = np.maximum(J_end, 0.0) * hh.keep_probability(P, choice_end)
    steps = [None] * T
    realised = {}                                            # what is needed to value the first `keep` dates as they turn out
    for t in range(T - 1, -1, -1):
        pr = hh.Prices(R=R[t], D=D[t], wage=wage[t], finding=finding[t], transfer=S[t] / mass)
        Vbar, policy = hh.income_stage(P, pr, V)
        J = hh.job_value_step(P, wage[t], lam[t], policy, R[t + 1], cont)
        pr.hire_ok = (J >= 0).astype(float); pr.layoff = (J < 0).astype(float)
        V, Tm, choice = hh.start_stage(P, pr, Vbar)
        cont = np.maximum(J, 0.0) * hh.keep_probability(P, choice)
        if detail and keep is not None and t < keep:        # the policies, the status probabilities and the expected choice costs
            realised[t] = (policy, Tm, hh.apply_transitions(P, Tm, Vbar) - V)
        else:
            policy.pop('flow')
        steps[t] = dict(pr=pr, sol=dict(policy=policy, T=Tm, choice=choice), J=J)
    V_start = V                                              # values at the first date under the expectations of this path

    # ---- forward: distribution and household aggregates ----------------------
    mu = state['mu']; kept = {}
    ags = []
    kn = P.home == N
    track = dict(status=np.zeros((T, 2, P.ns)), layoffs=np.zeros((T, 2)), access=np.zeros(T)) if detail else None
    if detail:
        cohort, groups = state['cohort'], state['groups']
        track['cohort'] = np.zeros((T, P.ns)); track['cohort_unable'] = np.zeros((T, P.ns))
        track['groups'] = np.zeros((T, len(groups), 2, P.ns))
        track['groups_consumption'] = np.zeros((T, len(groups))); track['groups_assets'] = np.zeros((T, len(groups)))
    for t in range(T):
        st = steps[t]
        mui = hh.income_distribution(P, mu, st['sol']['T'])
        ags.append(hh.aggregates(P, st['pr'], st['sol'], mu, mui, st['J']))
        if detail:
            for g in range(2):
                track['status'][t, g] = mui[P.home == g].sum((0, 1, 3))
            track['access'][t] = mui[kn][:, 1].sum() / mui[kn].sum()
            for h in range(P.nh):
                stay = 1 - P.exit_E[h].sum()
                for k in range(2):
                    for g in range(2):
                        if hh.accessible(P, h, k, g):
                            take = st['sol']['choice'][(h, k)].get(('take', g), 0.0)
                            track['layoffs'][t, g] += (stay * (1 - take) * (1 - P.separation[g])
                                                       * st['pr'].layoff[h, k, g] * mu[h, k, hh.emp(g)]).sum()
            ci = hh.income_distribution(P, cohort, st['sol']['T'])
            track['cohort'][t] = ci.sum((0, 1, 3))
            track['cohort_unable'][t] = ci[(P.gamma == 0) & kn][:, 0].sum((0, 2))
            cohort = hh.advance(P, st['sol']['policy'], ci, entrants=False)
            groups = follow_groups(P, groups, st['sol'], t, track)
        mu = hh.advance(P, st['sol']['policy'], mui)
        if detail and keep is not None and t + 1 == keep:
            kept = dict(mu=mu, cohort=cohort, groups=groups)
    col = lambda key: np.array([a[key] for a in ags])
    L = col('labour')

    # ---- production: capital paths given labour and returns ------------------
    start = tuple(state['K'])
    if end is None:                                          # labour stays at its last value
        L_end = L[-1]
        old = guess.get('K')
        u_end = max(float(u[-1]), 1.0) if purchases else float(u[-1])
        K_end = prod.stationary_capital(tech, frontier[T], R[T], L_end[N], L_end[M], start if old is None else old[-1],
                                        u=u_end ** invest_weight if returns == 'policy' else 1.0)
    else:
        L_end, K_end = end['households']['ag']['labour'], (end['allocation'].KC, end['allocation'].KA)
    recruit = (P.tightness(finding) * col('applications')) @ eco.vacancy_cost
    receipts = col('receipts')
    levy = receipts[:, 1] + receipts[:, 2] + col('tuition_subsidy') + col('allowance') + col('transfer')
    tuition = P.tuition * col('instruction')
    # deductions from output in capitalists' income, apart from the upkeep of capital
    charges_fixed = col('payroll') + recruit + receipts[:, 0] + levy + col('claims') * D + (R[:T] - 1) * col('assets_open')
    policy = None
    if mpc is not None and returns == 'policy':
        policy = dict(floor=eco.R, mpc=mpc, base=base_ref if anchor is None else anchor0,
                      anchor_speed=anchor_speed, base_path=anchor, delayed=delayed, ref=base_ref, purchases=purchases,
                      other=col('consumption') + tuition + recruit, charges=charges_fixed, u=u, weight=invest_weight,
                      response=rate_response, feedback=feedback, payroll=wage * L, lam=lam,
                      lower=eco.R if rate_lower is None else rate_lower)
    cap = prod.capital_path(tech, frontier, R, np.r_[L[:, N], L_end[N]], np.r_[L[:, M], L_end[M]], start, K_end,
                            guess=guess.get('K'), policy=policy, I_before=state['I'])
    q = cap['q']
    G = cap['G'] if policy is not None else np.zeros(T)      # public purchases (zero unless `purchases`)
    if policy is not None:                                   # capacity use solved with the capital paths
        u_guess, u = u, cap['u']
    lam_new = u[:, None] * np.column_stack([q['lamN'][:T], q['lamM'][:T]])      # what employment accounts receive

    # ---- free entry -----------------------------------------------------------
    value = col('job_value_per_application')
    q_target = np.where(value > 0, eco.vacancy_cost / np.maximum(value, 1e-300), np.inf)
    finding_new = np.clip(P.match_eff - q_target, 0.0, P.match_eff - 1e-9)

    # ---- accounts -------------------------------------------------------------
    capacity = q['Y'][:T]
    Y = u * capacity
    invest, install = cap['goods_investment'], cap['Phi'].sum(1)
    F = Y - col('payroll') - invest - install - recruit - receipts[:, 0] - levy - G
    D_new = np.zeros(T); advances = np.zeros(T + 1); advances[0] = state['advances']
    for t in range(T):
        due = R[t] * advances[t]
        D_new[t] = max(F[t] - due, 0.0); advances[t + 1] = max(due - F[t], 0.0)
    C_cap = F - col('claims') * D_new + col('saving') - R[:T] * col('assets_open')
    uses = col('consumption') + C_cap + tuition + invest + install + recruit + G
    u_new, saving_C = np.ones(T), np.zeros(T)
    R_new = R.copy()
    if mpc is not None:
        dep = np.array([tech.dep_C, tech.dep_A]); kap = np.array([tech.install_C, tech.install_A])
        K = cap['K'][:T]
        upkeep = dep * K                                                        # replacement investment
        upkeep_goods = upkeep[:, 0] + (0.0 if tech.ai_sector else upkeep[:, 1])
        upkeep_install = (kap * upkeep ** 2 / (2 * (K + tech.install_base))).sum(1)
        charges = charges_fixed + upkeep_goods + upkeep_install + G
        base = reference_income(eco, ref)
        other = col('consumption') + tuition + invest + install + recruit      # spending by everyone else
        demand = (other + (1 - mpc) * base - mpc * charges) / (1 - mpc)        # output at which the rule holds
        excess = demand / capacity
        u_new = np.clip(excess, 0.2, 1.0 if returns == 'policy' else 2.0)
        if returns == 'policy':                                                 # solved with the capital paths
            R_new, u_new, u = cap['R'].copy(), cap['ugen'], u_guess
        saving_C = (Y - charges) - C_cap                                        # capitalists' income less consumption
    income_C = Y - charges if mpc is not None else np.zeros(T)
    anchor_new = np.full(T, anchor0)
    if anchor is not None:                                   # the anchor follows capitalists' income
        anchor = cap['base']                                 # the path used in the spending rule
        for t in range(T - 1):
            anchor_new[t + 1] = anchor_new[t] + anchor_speed * (income_C[t] - anchor_new[t])
    if returns == 'euler':
        C_end = C_cap[-1] if end is None else end['accounts']['capitalist_consumption']
        path = np.r_[np.maximum(C_cap, 1e-3 * C_end), C_end]
        R_new[1:] = path[1:] / (eco.beta_C * path[:-1])

    gaps = dict(lam=np.log(lam_new / lam), finding=finding_new - finding, D=(D_new - D) / Y[0],
                R=np.log(R_new / R), u=u_new - u)
    use = cap['u'] if policy is not None else (u_new if mpc is not None else u)      # capacity use on this path
    licence = (1 - use * tech.alpha_A) * q['pA'][:T] * cap['I'][:, 1]                 # licence rent (Section "Demand")
    S_new = S
    if transfer_rate is not None:                            # a share of the rise in the licence rent goes to residents
        licence_ref = (1 - tech.alpha_A) * tech.dep_A * ref['allocation'].KA * ref['allocation'].pA
        S_new = transfer_rate * (licence - licence_ref)
        gaps['S'] = (S_new - S) / ref['accounts']['output']
    if anchor is not None:                                   # a check: zero once the paths are consistent
        gaps['anchor'] = (anchor_new - anchor) / base_ref
    out = dict(implied=dict(lam=lam_new, finding=finding_new, D=D_new, R=R_new, K=cap['K'], V_end=V_end, u=u_new, S=S_new,
                            anchor=anchor if anchor is not None else anchor_new), gaps=gaps,
               error={k: float(np.max(np.abs(v))) for k, v in gaps.items()},
               capital_residual=cap['residual'], resource_gap=float(np.max(np.abs(Y - uses) / Y)))
    if detail:
        out['welfare'] = dict(V0=V_start, dates=[realised[t] for t in sorted(realised)])
    if kept:
        out['state'] = dict(kept, K=cap['K'][keep].copy(), wage=wage[keep - 1].copy(), I=cap['I'][keep - 1].copy(),
                            advances=float(advances[keep]),
                            anchor=float((anchor if anchor is not None else anchor_new)[keep]))
    if detail:
        out['paths'] = dict(
            frontier=frontier[:T], automated=q['m'][:T], wage=wage, lam=lam, finding=finding, D=D, R=R[:T],
            employment=col('employment'), unemployment=col('unemployment'), outside=col('outside'), labour=L,
            output=Y, capacity=capacity, utilisation=use, capitalist_saving=saving_C, payroll=col('payroll'),
            transfer=col('transfer'), purchases=G, licence_rent=licence,
            household_consumption=col('consumption'), capitalist_consumption=C_cap, capitalist_income=income_C,
            anchor=anchor if anchor is not None else np.full(T, base_ref),
            investment=cap['I'], capital=cap['K'][:T], ai_price=q['pA'][:T], operating_cash=F, advances=advances[:T], recruit=recruit,
            participants=col('participants'), qualified=col('qualified'), instruction=col('instruction'),
            assets=col('assets_open'), status_by_home=track['status'], layoffs=track['layoffs'],
            knowledge_home_access=track['access'], cohort=track['cohort'], cohort_unable=track['cohort_unable'],
            cohort_groups=track['groups'], cohort_groups_consumption=track['groups_consumption'],
            cohort_groups_assets=track['groups_assets'])
    return out


def reference_cohort(eco, ref, T):
    """Statuses of the initial knowledge cohort if the reference continued."""
    P = eco.P
    sol, mu = ref['households']['sol'], ref['households']['mu']
    cohort = np.zeros_like(mu)
    cohort[:, :, [hh.emp(N), hh.unemp(N)]] = mu[:, :, [hh.emp(N), hh.unemp(N)]]
    out = np.zeros((T, P.ns))
    for t in range(T):
        ci = hh.income_distribution(P, cohort, sol['T'])
        out[t] = ci.sum((0, 1, 3))
        cohort = hh.advance(P, sol['policy'], ci, entrants=False)
    return out


def reference_groups(eco, ref, T):
    """The groups of the initial knowledge cohort (see cohort_groups) if the reference continued."""
    P = eco.P
    sol, mu = ref['households']['sol'], ref['households']['mu']
    cohort = np.zeros_like(mu)
    cohort[:, :, [hh.emp(N), hh.unemp(N)]] = mu[:, :, [hh.emp(N), hh.unemp(N)]]
    groups = cohort_groups(P, cohort)
    track = dict(groups=np.zeros((T, len(groups), 2, P.ns)), groups_consumption=np.zeros((T, len(groups))),
                 groups_assets=np.zeros((T, len(groups))))
    for t in range(T):
        groups = follow_groups(P, groups, sol, t, track)
    return track


def solve(eco, ref, end, frontier, phi, guess=None, returns='fixed', step=0.05, burn=12, memory=6, tol=1e-3,
          max_iter=150, verbose=False, mpc=None, invest_weight=1.0, rate_response=0.125, state=None, keep=None,
          feedback=0.7, rate_lower=None, anchor_speed=None, delayed=False, transfer_rate=None, purchases=False):
    """Fixed point of the path map by damped iteration with acceleration.

    Workers qualified for both occupations move quickly when relative pay changes,
    so the implied prices respond strongly to the guessed prices and the price step
    must be small (about 0.05). Job finding and payouts take larger steps. After
    `burn` plain passes the last `memory` passes are used to extrapolate (Anderson
    acceleration); if the gap rises, the extrapolation is dropped for `burn` passes.
    The best iterate is kept, so the result is the closest candidate even if `tol`
    is not reached.
    """
    guess = initial_guess(ref, end, frontier, eco.R) if guess is None else dict(guess)
    T = len(guess['D']); Y0 = ref['accounts']['output']
    if guess.get('u') is None:
        guess['u'] = np.ones(T)
    moving = anchor_speed is not None                        # the anchor path is carried as a starting value only
    base_ref = reference_income(eco, ref)
    if moving and guess.get('anchor') is None:
        guess['anchor'] = np.full(T, base_ref if state is None else state.get('anchor', base_ref))
    taxed = transfer_rate is not None                        # the transfer path is carried in the guess and must reproduce itself
    if taxed and guess.get('S') is None:
        guess['S'] = np.zeros(T)
    scale = np.r_[np.full(2 * T, step), np.full(2 * T, 4 * step), np.full(T, 6 * step),
                  np.full(T, 6 * step if returns == 'policy' else step), np.full(T, 10 * step if returns == 'policy' else 4 * step)]
    scale = np.minimum(scale, 1.0)                           # never move further than the implied value
    cap = 2.0

    def pack(g):
        return np.r_[np.log(g['lam']).ravel(), g['finding'].ravel(), g['D'] / Y0, np.log(g['R'][1:]), g['u']]

    def unpack(x, K, V_end):
        R = guess['R'].copy(); R[1:] = np.exp(x[5 * T:6 * T])
        if returns == 'policy':
            R[1:] = np.clip(R[1:], eco.R if rate_lower is None else rate_lower, 1.5 * eco.R)   # never below the floor
        new = dict(lam=np.exp(x[:2 * T]).reshape(T, 2),
                   finding=np.clip(x[2 * T:4 * T].reshape(T, 2), 0.0, eco.P.match_eff - 1e-9),
                   D=np.maximum(x[4 * T:5 * T], 0.0) * Y0, R=R, K=K, V_end=V_end, u=np.clip(x[6 * T:7 * T], 0.2, cap))
        if moving:
            new['anchor'] = guess['anchor']
        if taxed:
            new['S'] = guess['S']
        return new

    x = pack(guess)
    X, Rs, plain = [], [], burn
    best, best_err, last = None, np.inf, np.inf
    for it in range(max_iter):
        out = evaluate(eco, ref, end, frontier, guess, phi, returns, mpc=mpc, invest_weight=invest_weight,
                       rate_response=rate_response, state=state, feedback=feedback, rate_lower=rate_lower,
                       anchor_speed=anchor_speed, delayed=delayed, transfer_rate=transfer_rate, purchases=purchases)
        g = out['gaps']
        r = np.r_[g['lam'].ravel(), g['finding'].ravel(), g['D'], g['R'][1:], g['u']]
        r = r * scale
        if moving:
            guess['anchor'] = out['implied']['anchor']
        err = max(out['error'].values())
        if verbose:
            print(f"  pass {it:3d}: gaps price {out['error']['lam']:.2e} finding {out['error']['finding']:.2e} "
                  f"payout {out['error']['D']:.2e} return {out['error']['R']:.2e} demand {out['error']['u']:.2e}"
                  f"{'' if plain > 0 else '  accelerated'}", flush=True)
        if err < best_err:
            best, best_err = dict(guess, K=out['implied']['K'], V_end=out['implied']['V_end']), err
        if err < tol:
            break
        if plain <= 0 and err > 1.5 * last:              # extrapolation failed: back to the best iterate
            guess = dict(best); x = pack(guess); X, Rs, plain, last = [], [], burn, np.inf
            continue
        last = err
        X.append(x.copy()); Rs.append(r.copy())
        X, Rs = X[-memory - 1:], Rs[-memory - 1:]
        if plain <= 0 and len(X) > 2:
            dX = np.diff(np.array(X), axis=0).T; dR = np.diff(np.array(Rs), axis=0).T
            gamma = np.linalg.lstsq(dR, r, rcond=1e-6)[0]
            x = x + r - (dX + dR) @ gamma
        else:
            x = x + r
        plain -= 1
        guess = unpack(x, out['implied']['K'], out['implied']['V_end']); x = pack(guess)
        if taxed:
            guess['S'] = out['implied']['S']                 # the transfer implied by this pass is what residents receive in the next
    final = evaluate(eco, ref, end, frontier, best, phi, returns, detail=True, mpc=mpc, invest_weight=invest_weight,
                     rate_response=rate_response, state=state, keep=keep, feedback=feedback, rate_lower=rate_lower,
                     anchor_speed=anchor_speed, delayed=delayed, transfer_rate=transfer_rate, purchases=purchases)
    final.update(guess=best, passes=it + 1, max_gap=max(final['error'].values()),
                 converged=bool(max(final['error'].values()) < tol))
    return final


def shifted(guess, n, T):
    """Starting paths for a path that begins n dates later and has T dates."""
    def move(a, length):
        a = a[n:]
        if len(a) >= length:
            return a[:length].copy()
        return np.concatenate([a, np.repeat(a[-1:], length - len(a), axis=0)])
    new = dict(lam=move(guess['lam'], T), finding=move(guess['finding'], T), D=move(guess['D'], T),
               u=move(guess['u'], T), R=move(guess['R'], T + 1), K=move(guess['K'], T + 1), V_end=guess['V_end'])
    if guess.get('anchor') is not None:
        new['anchor'] = move(guess['anchor'], T)
    if guess.get('S') is not None:
        new['S'] = move(guess['S'], T)
    return new


def solve_surprises(eco, ref, levels, quarters, final_level, T, phi, tol=1e-3, max_iter=80, final_iter=200, verbose=False,
                    slump_start=None, step_horizon=100, start=None, checkpoint=None, retries=2, **kw):
    """Capability rises in steps that are not foreseen.

    During step i (of `quarters` dates) the frontier is levels[i] and everyone expects
    it to stay there; the next step is a surprise. Each step is the first `quarters`
    dates of the perfect-foresight path that starts from the current state with a
    constant frontier. After the last step the frontier stays at `final_level`, and
    expectations are correct from then on. `start` is the state at the first date (the
    reference if omitted). Returns the realised paths over T dates and a report for each step.

    A step that does not reach `tol` is solved again from the same starting paths with half the solver step (and at
    least twelve earlier passes in the extrapolation), up to `retries` times; this changes the route of the solver and
    not the conditions that the paths satisfy. With `checkpoint` (a file name) the state is saved after every step
    that reaches `tol`, and a run that finds the file continues after the last saved step.
    """
    state, guess, pieces, report = (start_state(eco, ref) if start is None else start), None, [], []
    values = []                                              # per step: values at its first date and what its retained dates need
    steps = [(level, quarters) for level in levels] + [(final_level, None)]
    used, first = 0, 0
    if checkpoint is not None and os.path.exists(checkpoint):
        saved = pickle.load(open(checkpoint, 'rb'))
        state, guess, pieces, report, used, first = saved[:6]
        values = saved[6] if len(saved) > 6 else None        # a checkpoint written before welfare was added
    for i, (level, n) in enumerate(steps):
        if i < first:
            continue
        last = n is None
        horizon = T - used if last else max(step_horizon, n + 80)
        frontier = np.full(horizon + 1, level)
        if slump_start is not None:                          # test of uniqueness: start every step from a slump
            guess = initial_guess(ref, None, frontier, eco.R) if guess is None else dict(guess)
            guess['u'] = np.full(horizon, slump_start); guess['lam'] = guess['lam'] * slump_start
            guess['R'] = guess['R'].copy(); guess['R'][1:] = eco.R if kw.get('rate_lower') is None else kw['rate_lower']
        attempt, passes = dict(kw), 0
        for trial in range(retries + 1):
            out = solve(eco, ref, None, frontier, phi, guess=guess, tol=tol, max_iter=final_iter if last else max_iter,
                        verbose=verbose, state=state, keep=None if last else n, **attempt)
            passes += out['passes']; solver_step = attempt.get('step', 0.05)
            if out['max_gap'] < tol or trial == retries:
                break
            attempt = dict(attempt, step=solver_step / 2, memory=max(attempt.get('memory', 6), 12))
            if verbose:
                print(f"step {i}: largest gap {out['max_gap']:.1e} after {out['passes']} passes; solved again with solver step {solver_step / 2}", flush=True)
        take = horizon if last else n
        pieces.append({k: v[:take] for k, v in out['paths'].items()})
        if values is not None:
            values.append(out['welfare'])
        report.append(dict(step=i, frontier=float(level), start=used, passes=passes, max_gap=out['max_gap'],
                           resource_gap=out['resource_gap'], solver_step=solver_step))
        if verbose:
            print(f"step {i}: frontier {level:.3f} from quarter {used}: {out['passes']} passes, largest gap {out['max_gap']:.1e}", flush=True)
        if last:
            break
        state = out['state']; used += n
        nxt = T - used if i + 1 == len(steps) - 1 else max(step_horizon, n + 80)
        guess = shifted(out['guess'], n, nxt)
        if checkpoint is not None and out['max_gap'] < tol:
            pickle.dump((state, guess, pieces, report, used, i + 1, values), open(checkpoint, 'wb'))
    paths = {k: np.concatenate([p[k] for p in pieces], axis=0) for k in pieces[0]}
    # Welfare as it turns out. After the last step expectations are correct, so the values at its first date are exact.
    # Before it, residents act on the expectations of each step, and the dates that are retained are valued with the
    # policies and status probabilities of that step and with the values that actually follow.
    welfare = None
    if values is not None and len(values) == len(steps):
        welfare = values[-1]['V0']
        for step in reversed(values[:-1]):
            for policy, Tm, cost in reversed(step['dates']):
                welfare = hh.apply_transitions(eco.P, Tm, hh.policy_value(eco.P, policy, welfare)) - cost
    return dict(welfare_start=welfare, welfare_expected=None if values is None else values[0]['V0'],
                welfare_steps=None if welfare is None else values[:-1], paths=paths, steps=report, passes=sum(r['passes'] for r in report), max_gap=max(r['max_gap'] for r in report),
                resource_gap=max(r['resource_gap'] for r in report), converged=all(r['max_gap'] < tol for r in report))
