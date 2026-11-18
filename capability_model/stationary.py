"""Stationary equilibrium (Section 4.7 of the paper).

`reference` calibrates the technology and recruiting costs so that the observed
wages and job-finding rates are an equilibrium. `solve` finds the stationary
equilibrium for another capability frontier or transition policy.

In a stationary equilibrium the posted wage equals its flexible target, so the
gradual wage rule and layoffs play no role here; they matter on transition paths.

AI capital is bought within the business sector and the scarce input belongs to it,
so neither appears in operating cash: only conventional investment uses ordinary goods.
"""
from dataclasses import dataclass, replace
import math
import numpy as np
from . import households as hh
from . import production as prod


@dataclass
class Economy:
    P: hh.HouseholdParams
    tech: prod.Technology
    vacancy_cost: np.ndarray
    wage_share: float
    beta_C: float

    @property
    def R(self):
        return 1 / self.beta_C


def manual_price(eco, lamN, LM, KC, KA):
    """Manual service price at which price equals the markup over unit cost."""
    lo, hi = 1e-6, 1e3
    def gap(x):
        return prod.allocate(eco.tech, eco.R, lamN, x, LM, KC, KA).price_gap
    if gap(lo) > 0:
        return None                       # knowledge price too high: no positive manual price
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if gap(mid) < 0 else (lo, mid)
    return math.sqrt(lo * hi)


def evaluate(P, R, wage_share, lam, finding, D, V0=None):
    """Household side at given service prices, job finding and payouts."""
    pr = hh.Prices(R=R, D=D, wage=wage_share * np.asarray(lam), finding=np.asarray(finding, float))
    sol = hh.solve_values(P, pr, V0)
    mu, mui = hh.stationary_distribution(P, sol)
    J = hh.job_values(P, pr, sol, np.asarray(lam), R)
    ag = hh.aggregates(P, pr, sol, mu, mui, J)
    return dict(prices=pr, sol=sol, mu=mu, mu_income=mui, J=J, ag=ag)


def accounts(eco, hhres, alloc, lam, finding):
    """Operating cash, payouts, capitalist consumption and the resource check."""
    P, ag = eco.P, hhres['ag']
    theta = P.tightness(finding)
    vacancies = theta * ag['applications']
    recruit = float(eco.vacancy_cost @ vacancies)
    private, government, support = ag['receipts']
    transition = ag['tuition_subsidy'] + ag['allowance']
    levy = government + support + transition
    F = alloc.Y - ag['payroll'] - alloc.investment - alloc.installation - recruit - private - levy
    D = max(F, 0.0)
    R = eco.R
    C_cap = F - ag['claims'] * D + ag['saving'] - R * ag['assets_open']
    tuition = P.tuition * ag['instruction']
    uses = ag['consumption'] + C_cap + tuition + alloc.investment + alloc.installation + recruit
    return dict(output=alloc.Y, gdp=alloc.Y - recruit + alloc.ai_value, recruit=recruit, levy=levy, operating_cash=F, payouts=D,
                capitalist_consumption=C_cap, household_consumption=ag['consumption'], tuition=tuition,
                investment=alloc.investment, installation=alloc.installation,
                ai_investment_value=alloc.ai_value, scarce_rent=alloc.scarce_rent, ai_price=alloc.pA,
                resource_gap=(alloc.Y - uses) / alloc.Y, vacancies=vacancies)


def reference(P, wage_ref, wage_share, beta_C, technology, gdp_target=None, tol=1e-10):
    """Calibrate technology and recruiting costs at the observed wages and job finding.

    `technology` holds the arguments of production.calibrate other than prices and labour.
    If `gdp_target` is given, the markup is chosen so that reference value added
    (output less recruiting) equals it; otherwise the given variety elasticity is used.
    """
    R = 1 / beta_C
    lam = np.asarray(wage_ref) / wage_share
    finding = P.finding_ref
    technology = dict(technology)
    D, V, ai_value = 0.0, None, 0.0
    for it in range(50):
        res = evaluate(P, R, wage_share, lam, finding, D, V)
        V = res['sol']['V']
        L = res['ag']['labour']
        cost = P.filling(P.tightness(finding)) * res['ag']['job_value_per_application']
        if gdp_target is not None:
            recruit = float(cost @ (P.tightness(finding) * res['ag']['applications']))
            bundle = lam[N_] * L[N_] / (1 - technology['frontier']) + lam[M_] * L[M_]
            mc = bundle / ((1 - technology['alpha']) * (gdp_target + recruit - ai_value))   # value added includes the AI good
            assert 0 < mc < 1, mc
            technology['variety_elasticity'] = 1 / (1 - mc)
        tech, alloc = prod.calibrate(R=R, lamN=lam[N_], lamM=lam[M_], LN=L[N_], LM=L[M_], **technology)
        eco = Economy(P, tech, cost, wage_share, beta_C)
        acc = accounts(eco, res, alloc, lam, finding)
        if abs(acc['payouts'] - D) < tol and abs(alloc.ai_value - ai_value) < tol:
            break
        D, ai_value = acc['payouts'], alloc.ai_value
    return eco, dict(lam=lam, finding=np.array(finding), D=D, households=res, allocation=alloc, accounts=acc,
                     residuals=dict(labour_N=0.0, price=alloc.price_gap, entry=np.zeros(2)), iterations=it + 1)


N_, M_ = hh.N, hh.M


def solve(eco, start, tol=1e-6, accept=1e-4, newton=20, jacobian=None, verbose=False):
    """Stationary equilibrium of `eco`, starting from the state of a nearby equilibrium.

    Unknowns: the knowledge service price and the two job-finding probabilities.
    Conditions: knowledge labour clears and free entry holds in both occupations.
    The manual service price follows from pricing, and payouts from the accounts.
    Newton steps with a Broyden-updated Jacobian. Household choices are discrete on
    a grid, so residuals are not perfectly smooth; `accept` is the largest residual
    reported as converged.
    """
    P, R = eco.P, eco.R
    state = dict(V=start['households']['sol']['V'], KC=start['allocation'].KC, KA=start['allocation'].KA,
                 LM=float(start['households']['ag']['labour'][M_]), D=float(start['D']))
    count = [0]

    def f(x):
        lamN = math.exp(x[0]); finding = np.clip(x[1:], 0.0, P.match_eff - 1e-9)
        lamM = manual_price(eco, lamN, state['LM'], state['KC'], state['KA'])
        if lamM is None:
            return None, None
        lam = np.array([lamN, lamM])
        res = evaluate(P, R, eco.wage_share, lam, finding, state['D'], state['V'])
        count[0] += 1
        ag = res['ag']
        alloc = prod.allocate(eco.tech, R, lamN, lamM, float(ag['labour'][M_]), state['KC'], state['KA'])
        acc = accounts(eco, res, alloc, lam, finding)
        value = ag['job_value_per_application']
        q_target = np.where(value > 0, eco.vacancy_cost / np.maximum(value, 1e-300), np.inf)
        entry_gap = np.clip(P.match_eff - q_target, 0.0, None) - finding
        excess = math.log(ag['labour'][N_] / alloc.labour_demand_N) if ag['labour'][N_] > 0 else -50.0
        out = dict(lam=lam, finding=finding, D=acc['payouts'], households=res, allocation=alloc, accounts=acc,
                   residuals=dict(labour_N=excess, price=alloc.price_gap, entry=entry_gap))
        return np.array([excess, entry_gap[0], entry_gap[1]]), out

    def commit(out):
        ag, alloc = out['households']['ag'], out['allocation']
        state.update(V=out['households']['sol']['V'], KC=alloc.KC, KA=max(alloc.KA, 1e-12),
                     LM=float(ag['labour'][M_]), D=out['D'])

    def fd(x, r):
        Jm = np.zeros((3, 3))
        for c in range(3):
            xp = x.copy(); xp[c] += 5e-4
            rp, _ = f(xp)
            Jm[:, c] = (rp - r) / 5e-4
        return Jm

    x = np.array([math.log(start['lam'][N_]), start['finding'][0], start['finding'][1]], float)
    r, out = f(x)
    while r is None:
        x[0] += 0.1; r, out = f(x)
    commit(out)
    Jm, fresh = jacobian, False
    for it in range(newton):
        if verbose:
            print(f"  lamN {math.exp(x[0]):.5f} f {np.round(x[1:], 4)} residual {np.round(r, 7)}")
        if np.max(np.abs(r)) < tol:
            break
        if Jm is None:
            Jm, fresh = fd(x, r), True
        try:
            dx = np.linalg.solve(Jm, -r)
        except np.linalg.LinAlgError:
            Jm = None; continue
        scale, accepted = 1.0, False
        for _ in range(6):
            xn = x + scale * dx
            rn, outn = f(xn)
            if rn is not None and np.max(np.abs(rn)) < np.max(np.abs(r)):
                accepted = True; break
            scale *= 0.5
        if not accepted:
            if fresh:
                break                           # a fresh Jacobian gave no descent direction
            Jm = None; continue                 # retry with a fresh Jacobian
        s, y = xn - x, rn - r
        Jm = Jm + np.outer(y - Jm @ s, s) / (s @ s); fresh = False
        x, r, out = xn, rn, outn
        commit(out)
    out['iterations'] = count[0]
    out['max_residual'] = float(np.max(np.abs(r)))
    out['converged'] = bool(out['max_residual'] < accept)
    out['jacobian'] = Jm
    return out


def solve_path(eco, start, start_frontier, target, step=0.05, verbose=False):
    """Raise the frontier gradually from an equilibrium at `start_frontier` to `target`."""
    current, level, jac = start, start_frontier, start.get('jacobian')
    while level < target - 1e-12:
        s = min(step, target - level)
        while True:
            res = solve(with_frontier(eco, level + s), current, jacobian=jac)
            if res['converged'] or s < 0.004:
                break
            s /= 2; jac = None
        if verbose:
            print(f"    frontier {level + s:.4f}: residual {res['max_residual']:.1e} evaluations {res['iterations']}", flush=True)
        if not res['converged']:
            res['reached'] = level
            return res
        current, level, jac = res, level + s, res['jacobian']
    current['reached'] = level
    return current


def with_frontier(eco, frontier):
    return replace(eco, tech=replace(eco.tech, frontier=frontier))


def with_policy(eco, **policy):
    return replace(eco, P=replace(eco.P, **policy))


def summary(eco, res, base=None):
    """Headline numbers of an equilibrium; changes relative to `base` when given."""
    P = eco.P
    ag, alloc, acc = res['households']['ag'], res['allocation'], res['accounts']
    mui = res['households']['mu_income']
    out = dict(
        frontier=eco.tech.frontier, automated_share=alloc.m, ai_price=alloc.pA, ai_capital=alloc.KA,
        scarce_rent=alloc.scarce_rent, gdp=acc['gdp'], investment=acc['investment'] + acc['ai_investment_value'],
        knowledge_wage=float(eco.wage_share * res['lam'][N_]), manual_wage=float(eco.wage_share * res['lam'][M_]),
        knowledge_finding=float(res['finding'][N_]), manual_finding=float(res['finding'][M_]),
        knowledge_employment=float(ag['employment'][N_]), manual_employment=float(ag['employment'][M_]),
        unemployment=float(ag['unemployment'].sum()), outside=ag['outside'].tolist(),
        output=acc['output'], household_consumption=acc['household_consumption'],
        capitalist_consumption=acc['capitalist_consumption'], payouts=acc['payouts'],
        instruction=ag['instruction'], participants=ag['participants'], qualified=ag['qualified'],
        resource_gap=acc['resource_gap'], residual_labour=res['residuals']['labour_N'],
        residual_entry=np.asarray(res['residuals']['entry']).tolist())
    # where knowledge-home residents are
    kn = P.home == N_
    status = mui[kn].sum((0, 1, 3)); total = status.sum()
    out['knowledge_home'] = dict(
        knowledge_job=float(status[hh.emp(N_)] / total), manual_job=float(status[hh.emp(M_)] / total),
        unemployed=float((status[hh.unemp(N_)] + status[hh.unemp(M_)]) / total),
        outside={name: float(status[hh.PRIVATE + i] / total) for i, name in enumerate(P.status_names[hh.PRIVATE:])},
        with_manual_access=float(mui[kn][:, 1].sum() / total),
        zero_assets=float(mui[kn][..., 0].sum() / total))
    if base is not None:
        b = summary(eco, base)
        out['change'] = {k: 100 * (out[k] / b[k] - 1) for k in
                         ['knowledge_wage', 'manual_wage', 'knowledge_employment', 'manual_employment', 'output', 'gdp',
                          'household_consumption', 'capitalist_consumption', 'ai_price', 'ai_capital']}
    return out
