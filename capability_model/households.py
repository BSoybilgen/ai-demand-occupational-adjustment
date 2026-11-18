"""Households and occupational adjustment (Section 2 of the paper).

State of a resident: liquid claims a (grid), permanent type h, access k in {0,1}
to the non-home occupation, labour status s. Statuses 0..3 are private
(N employed, N unemployed, M employed, M unemployed); the rest are outside the
private wage market.

Within a quarter: boundary -> offers and job loss -> search -> income and
choices -> qualification and renewal. `start_stage` covers the first three,
`income_stage` the fourth, `transition_matrix` adds the fifth.
"""
from dataclasses import dataclass, field
import math
import numpy as np

_erf = np.vectorize(math.erf, otypes=[float])

N, M = 0, 1
PRIVATE = 4


def emp(g):
    return 2 * g


def unemp(g):
    return 2 * g + 1


@dataclass
class HouseholdParams:
    agrid: np.ndarray
    # permanent types
    home: np.ndarray            # (nh,) home occupation
    eff: np.ndarray             # (nh, 2) efficiency units in each occupation
    gamma: np.ndarray           # (nh,) qualification aptitude
    beta: np.ndarray            # (nh,)
    claim: np.ndarray           # (nh,) payout claim per resident
    mass: np.ndarray            # (nh,) population mass
    entry_access: np.ndarray    # (nh,) share of entrants qualified for both occupations
    entry_status: np.ndarray    # (nh, ns) status distribution of entrants
    receipts: np.ndarray        # (3, nh, ns) private nonwage, government-linked, support
    # preferences and renewal
    sigma: float
    renewal: float
    estate_survivor: float
    estate_entrant: float
    estate_zero: float
    # boundary flows at the reference
    exit_E: np.ndarray          # (nh, n_out)
    exit_U: np.ndarray          # (nh, n_out)
    entry_E: np.ndarray         # (nh, n_out, 2) employed entry by occupation
    entry_U: np.ndarray         # (nh, n_out) entry to home unemployment
    outside_move: np.ndarray    # (nh, n_out, n_out) moves among outside statuses (diagonal ignored)
    entry_elasticity: float
    # matching and job choices
    separation: np.ndarray      # (2,)
    finding_ref: np.ndarray     # (2,)
    match_eff: np.ndarray       # (2,)
    cost_redirect: float
    cost_offer: float
    offer_intensity: np.ndarray  # (2,) on-the-job application probability, by current occupation
    search_intensity: np.ndarray  # (2,) searcher's application probability to the other occupation, by primary market
    # instruction
    tuition: float
    course_min: float
    course_cap: float
    kappa1: float
    kappa2: float
    zeta_sd: float
    zeta_nodes: int = 7
    course_points: int = 6
    # transition policy
    training_open: bool = True
    subsidy: float = 0.0
    allowance: float = 0.0
    deferred_share: float = 0.0
    access_share: float = 1.0
    status_names: list = field(default_factory=list)

    @property
    def nh(self):
        return len(self.home)

    @property
    def ns(self):
        return self.entry_status.shape[1]

    @property
    def na(self):
        return len(self.agrid)

    @property
    def n_out(self):
        return self.ns - PRIVATE

    def tightness(self, f):
        """Inverse of f = e*theta/(1+theta) (matching curvature one)."""
        f = np.asarray(f, float)
        return f / np.maximum(self.match_eff - f, 1e-12)

    def finding(self, theta):
        theta = np.asarray(theta, float)
        return self.match_eff * theta / (1 + theta)

    def filling(self, theta):
        return self.match_eff / (1 + np.asarray(theta, float))

    def course_grid(self):
        if not self.training_open or self.course_cap <= 0:
            return np.array([0.0])
        return np.concatenate([[0.0], np.linspace(self.course_min, self.course_cap, self.course_points)])

    def zeta(self):
        x, w = np.polynomial.hermite_e.hermegauss(self.zeta_nodes)
        return np.exp(self.zeta_sd * x - self.zeta_sd ** 2 / 2), w / w.sum()


@dataclass
class Prices:
    R: float                    # gross return received on opening claims
    D: float                    # business payouts per unit of claim
    wage: np.ndarray            # (2,) posted wage per efficiency unit
    finding: np.ndarray         # (2,) job-finding probability
    hire_ok: np.ndarray = None  # (nh, 2, 2, na) 1 if a match has nonnegative value
    layoff: np.ndarray = None   # (nh, 2, 2, na) 1 if an existing match is dissolved
    transfer: float = 0.0       # receipt of every resident, per unit of mass (policy experiment; zero otherwise)


def utility(c, sigma):
    out = np.full(c.shape, -np.inf)
    ok = c > 0
    if abs(sigma - 1) < 1e-12:
        out[ok] = np.log(c[ok])
    else:
        out[ok] = (c[ok] ** (1 - sigma) - 1) / (1 - sigma)
    return out


def surplus(gain, mean_cost):
    """Expected surplus and take-up probability of an option with exponential cost."""
    g = np.maximum(gain, 0.0)
    p = 1 - np.exp(-g / mean_cost)
    return g - mean_cost * p, p


def zeta_shares(level, slope, sd):
    """Choice among courses when the inconvenience draw zeta is lognormal with mean one.

    Course c is worth level[c] - slope[c]*zeta. Returns, for each course and asset
    level, the probability that it is chosen and the part of E[zeta] that falls in
    its range. The integration is exact, so choices respond smoothly to prices.
    """
    n = len(slope)
    lower = np.zeros(level.shape); upper = np.full(level.shape, np.inf)
    for c in range(n):
        for j in range(n):
            if slope[j] < slope[c]:        # flatter alternative: c is better below the crossing
                upper[c] = np.minimum(upper[c], (level[c] - level[j]) / (slope[c] - slope[j]))
            elif slope[j] > slope[c]:      # steeper alternative: c is better above the crossing
                lower[c] = np.maximum(lower[c], (level[j] - level[c]) / (slope[j] - slope[c]))
    upper = np.maximum(upper, lower)
    with np.errstate(divide='ignore'):
        lo, hi = np.log(lower), np.log(upper)
    cdf = lambda x, shift: 0.5 * (1 + _erf((x + shift) / (sd * math.sqrt(2))))
    return cdf(hi, sd * sd / 2) - cdf(lo, sd * sd / 2), cdf(hi, -sd * sd / 2) - cdf(lo, -sd * sd / 2)


def lottery(agrid, x):
    """Indices and weights that place mass at x on the grid, preserving the mean."""
    x = np.clip(x, agrid[0], agrid[-1])
    hi = np.clip(np.searchsorted(agrid, x, side='right'), 1, len(agrid) - 1)
    lo = hi - 1
    w = (x - agrid[lo]) / (agrid[hi] - agrid[lo])
    return lo, hi, w


def between(x, j, w):
    """Value of x at the saving choice: the share w of the way from grid node j to node j+1."""
    return (1 - w) * x[j] + w * x[np.minimum(j + 1, len(x) - 1)]


def refine(a, cash, cv, sigma, node_value, j, floor=0.0):
    """Saving between grid nodes.

    `j` is the best node for each level of `cash` and `node_value` the value of every
    node. The continuation value `cv` is linear between nodes, so in each of the two
    intervals next to the best node the best saving has a closed form. Returns the
    value, the lower node of the chosen interval, the share of the way to the next
    node, and consumption. Choosing between nodes makes saving, and therefore
    spending at each date, respond smoothly to prices.
    """
    na = len(a); rows = np.arange(na)
    best = node_value[rows, j].copy(); lo = j.copy(); w = np.zeros(na)
    da = np.diff(a)
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        slope = np.diff(cv) / da
        for k in (j - 1, j):                                 # the interval below and the interval above the best node
            kk = np.clip(k, 0, na - 2)
            sl = slope[kk]
            ok = (k >= 0) & (k <= na - 2) & np.isfinite(sl) & (sl > 0) & (a[kk] >= floor - 1e-12)
            c = np.where(ok, sl, 1.0) ** (-1 / sigma)        # marginal utility equals the slope
            x = np.clip(cash - c, a[kk], a[kk + 1])
            share = (x - a[kk]) / da[kk]
            v = utility(cash - x, sigma) + cv[kk] + share * (cv[kk + 1] - cv[kk])
            better = ok & (v > best + 1e-13)
            best = np.where(better, v, best); lo = np.where(better, kk, lo); w = np.where(better, share, w)
    return best, lo, w, cash - between(a, lo, w)


def accessible(P, h, k, g):
    return k == 1 or g == P.home[h]


def eligible(P, h, k, s):
    """Paid instruction: knowledge-home, no manual access, unemployed after search."""
    return P.home[h] == N and k == 0 and s == unemp(N)


def income_stage(P, pr, Vnext):
    """Consumption, saving and instruction. Returns expected values and policies."""
    nh, ns, na, a = P.nh, P.ns, P.na, P.agrid
    Vbar = np.full((nh, 2, ns, na), -np.inf)
    save = np.zeros((nh, 2, ns, na), dtype=np.int64)       # lower grid node of the saving choice
    mix = np.zeros((nh, 2, ns, na))                        # share of the way to the next node
    cons = np.zeros((nh, 2, ns, na))
    flow = np.zeros((nh, 2, ns, na))
    courses = P.course_grid()
    train = {}
    lo, hi, w = lottery(a, P.estate_survivor * a)
    z = P.receipts.sum(0)
    for h in range(nh):
        disc = P.beta[h] * (1 - P.renewal)
        for k in range(2):
            cont = (1 - w) * Vnext[h, k][:, lo] + w * Vnext[h, k][:, hi]      # (ns, na') value of carrying a'
            for s in range(ns):
                if s < PRIVATE and not accessible(P, h, k, s // 2):
                    continue
                earn = pr.wage[s // 2] * P.eff[h, s // 2] if s < PRIVATE and s % 2 == 0 else 0.0
                cash = pr.R * a + P.claim[h] * pr.D + earn + z[h, s] + pr.transfer
                if eligible(P, h, k, s) and (len(courses) > 1 or P.allowance > 0):
                    cash = cash + P.allowance
                    qual = (1 - w) * Vnext[h, 1, unemp(M)][lo] + w * Vnext[h, 1, unemp(M)][hi]
                    best_v, best_a, best_w = [], [], []
                    for ell in courses:
                        fee = (1 - P.subsidy) * P.tuition * ell
                        pi = 1 - np.exp(-P.gamma[h] * ell)
                        floor = max(0.0, (1 - P.access_share) * P.deferred_share * P.allowance - fee)
                        val = utility(cash[:, None] - fee - a[None, :], P.sigma) \
                            + disc * ((1 - pi) * cont[s] + pi * qual)[None, :]
                        val[:, a < floor - 1e-12] = -np.inf
                        j = val.argmax(1)
                        v, jl, wl, _ = refine(a, cash - fee, disc * ((1 - pi) * cont[s] + pi * qual), P.sigma, val, j, floor)
                        best_v.append(v); best_a.append(jl); best_w.append(wl)
                    best_v, best_a, best_w = np.array(best_v), np.array(best_a), np.array(best_w)     # (courses, na)
                    slope = P.kappa1 * courses                                   # inconvenience per unit of zeta
                    fixed = P.kappa2 / 2 * courses[:, None] ** 2
                    level = np.where(np.isfinite(best_v), best_v, -1e30) - fixed
                    prob, part = zeta_shares(level, slope, P.zeta_sd)            # (courses, na)
                    chosen = prob > 0
                    Vbar[h, k, s] = (np.where(chosen, prob * level, 0.0) - slope[:, None] * part).sum(0)
                    c = cash[None, :] - (1 - P.subsidy) * P.tuition * courses[:, None] - between(a, best_a, best_w)
                    c = np.where(chosen, c, 1.0)                                 # unused alternatives
                    flow_c = np.where(chosen, utility(c, P.sigma) - fixed - slope[:, None] * part / np.maximum(prob, 1e-300), 0.0)
                    # one row per course: weight is the probability of choosing it at each asset level
                    train[(h, k, s)] = dict(course=np.repeat(courses[:, None], na, 1), save=best_a, mix=best_w,
                                            weight=prob, cons=c, flow=flow_c)
                    cons[h, k, s] = (prob * c).sum(0)
                    save[h, k, s] = best_a[0]
                else:
                    val = utility(cash[:, None] - a[None, :], P.sigma) + disc * cont[s][None, :]
                    j = val.argmax(1)
                    Vbar[h, k, s], save[h, k, s], mix[h, k, s], cons[h, k, s] = refine(a, cash, disc * cont[s], P.sigma, val, j)
                    flow[h, k, s] = utility(cons[h, k, s], P.sigma)
    return Vbar, dict(save=save, mix=mix, cons=cons, train=train, flow=flow)


def policy_value(P, policy, Vnext):
    """Expected income-stage values of a fixed policy (no maximisation)."""
    nh, ns, na, a = P.nh, P.ns, P.na, P.agrid
    Vbar = np.full((nh, 2, ns, na), -np.inf)
    lo, hi, w = lottery(a, P.estate_survivor * a)
    for h in range(nh):
        disc = P.beta[h] * (1 - P.renewal)
        for k in range(2):
            cont = (1 - w) * Vnext[h, k][:, lo] + w * Vnext[h, k][:, hi]
            for s in range(ns):
                if s < PRIVATE and not accessible(P, h, k, s // 2):
                    continue
                tr = policy['train'].get((h, k, s))
                if tr is None:
                    Vbar[h, k, s] = policy['flow'][h, k, s] + disc * between(cont[s], policy['save'][h, k, s], policy['mix'][h, k, s])
                else:
                    qual = (1 - w) * Vnext[h, 1, unemp(M)][lo] + w * Vnext[h, 1, unemp(M)][hi]
                    pi = 1 - np.exp(-P.gamma[h] * tr['course'])
                    value = tr['flow'] + disc * ((1 - pi) * between(cont[s], tr['save'], tr['mix'])
                                                 + pi * between(qual, tr['save'], tr['mix']))
                    Vbar[h, k, s] = (tr['weight'] * value).sum(0)
    return Vbar


def flows(P, pr, Vbar, h, k):
    """Start-of-quarter values and status probabilities for one (type, access).

    Returns V (ns, na) and T (ns, ns, na): probability of reaching income-stage
    status s' from start-of-quarter status s at each asset level.
    """
    ns, na, n_out = P.ns, P.na, P.n_out
    home = P.home[h]
    V = np.array(Vbar[h, k], copy=True)
    T = np.zeros((ns, ns, na))
    info = {}
    ones = np.ones(na)
    f = np.zeros((2, na)); sep = np.zeros((2, na))
    for g in range(2):
        ok = ones if pr.hire_ok is None else pr.hire_ok[h, k, g]
        f[g] = pr.finding[g] * ok
        lay = 0.0 if pr.layoff is None else pr.layoff[h, k, g]
        sep[g] = P.separation[g] + (1 - P.separation[g]) * lay
        info[('sep', g)] = sep[g]
    VU = {}; searchT = {}
    for g in range(2):
        if not accessible(P, h, k, g):
            continue
        Eg, Ug = Vbar[h, k, emp(g)], Vbar[h, k, unemp(g)]
        row = np.zeros((ns, na))
        if k == 1:
            plans = {}
            for primary in (g, 1 - g):
                second = 1 - primary
                base = f[primary] * Vbar[h, k, emp(primary)] + (1 - f[primary]) * Ug
                S2, p2 = surplus(Vbar[h, k, emp(second)] - base, P.cost_offer)
                arrive = P.search_intensity[primary] * f[second]
                take2 = arrive * p2
                r = np.zeros((ns, na))
                r[emp(second)] = take2
                r[emp(primary)] += (1 - take2) * f[primary]
                r[unemp(g)] = (1 - take2) * (1 - f[primary])
                plans[primary] = (base + arrive * S2, r)
                info[('second_take', g, primary)] = take2; info[('second_accept', g, primary)] = p2
            S, p = surplus(plans[1 - g][0] - plans[g][0], P.cost_redirect)
            VU[g] = plans[g][0] + S
            info[('redirect', g)] = p
            row = (1 - p)[None, :] * plans[g][1] + p[None, :] * plans[1 - g][1]
        else:
            VU[g] = f[g] * Eg + (1 - f[g]) * Ug
            row[emp(g)] = f[g]; row[unemp(g)] = 1 - f[g]
        searchT[g] = row
    out = slice(PRIVATE, ns)
    for g in range(2):
        if not accessible(P, h, k, g):
            # off-support repair: treat as unemployed at home
            T[emp(g), unemp(home)] = 1.0; T[unemp(g), unemp(home)] = 1.0
            V[emp(g)] = Vbar[h, k, unemp(home)]; V[unemp(g)] = Vbar[h, k, unemp(home)]
            continue
        stay_E, stay_U = 1 - P.exit_E[h].sum(), 1 - P.exit_U[h].sum()
        VE0 = (1 - sep[g]) * Vbar[h, k, emp(g)] + sep[g] * VU[g]
        rowE = sep[g][None, :] * searchT[g]
        rowE[emp(g)] += 1 - sep[g]
        VE = VE0
        if k == 1:
            o = 1 - g
            S, p = surplus(Vbar[h, k, emp(o)] - VE0, P.cost_offer)
            take = P.offer_intensity[g] * f[o] * p
            info[('accept', g)] = p; info[('take', g)] = take
            VE = VE0 + P.offer_intensity[g] * f[o] * S
            rowE = (1 - take)[None, :] * rowE
            rowE[emp(o)] += take
        V[emp(g)] = P.exit_E[h] @ Vbar[h, k, out] + stay_E * VE
        V[unemp(g)] = P.exit_U[h] @ Vbar[h, k, out] + stay_U * VU[g]
        T[emp(g)] = stay_E * rowE; T[unemp(g)] = stay_U * searchT[g]
        T[emp(g), out] += P.exit_E[h][:, None]; T[unemp(g), out] += P.exit_U[h][:, None]
    scale = (pr.finding / P.finding_ref) ** P.entry_elasticity
    for o in range(n_out):
        s = PRIVATE + o
        base = P.entry_E[h, o].copy()
        if k == 0:                                   # without access, all entry is to the home occupation
            base = np.where(np.arange(2) == home, base.sum(), 0.0)
        row = np.zeros((ns, na))
        for g in range(2):
            ok = ones if pr.hire_ok is None else pr.hire_ok[h, k, g]
            row[emp(g)] = base[g] * scale[g] * ok
        row[unemp(home)] = P.entry_U[h, o]
        for o2 in range(n_out):
            if o2 != o:
                row[PRIVATE + o2] = P.outside_move[h, o, o2]
        row[s] = 1 - row.sum(0)
        assert row[s].min() > -1e-12, 'outside entry exceeds probability capacity'
        T[s] = row
        V[s] = np.where(row > 0, row * np.where(row > 0, Vbar[h, k], 0.0), 0.0).sum(0)
    return V, T, info


def start_stage(P, pr, Vbar):
    V = np.empty_like(Vbar); T = {}; choice = {}
    for h in range(P.nh):
        for k in range(2):
            V[h, k], T[(h, k)], choice[(h, k)] = flows(P, pr, Vbar, h, k)
    return V, T, choice


def apply_transitions(P, T, Vbar):
    """Start-of-quarter expectation of the income-stage values Vbar under the status probabilities T.

    The values of the start stage (flows) equal this expectation less the expected costs of the choices made there
    (redirecting search, accepting an offer), which do not depend on the values that follow."""
    V = np.zeros_like(Vbar)
    for h in range(P.nh):
        for k in range(2):
            V[h, k] = np.einsum('sta,ta->sa', T[(h, k)], np.where(np.isfinite(Vbar[h, k]), Vbar[h, k], 0.0))
    return V


def consumption_part(P, sol, tol=1e-11, max_iter=20000):
    """The part of the stationary values `sol` that comes from consumption: the expected discounted sum of
    c**(1 - sigma) / (1 - sigma) under the stationary policies, by start-of-quarter state. Used for welfare: if
    consumption is multiplied by (1 + x) at every date and in every state, this part is multiplied by
    (1 + x)**(1 - sigma) and the rest of the value is unchanged. Requires sigma different from one."""
    assert abs(P.sigma - 1) > 1e-9
    power = lambda c: np.where(c > 0, np.maximum(c, 1e-300) ** (1 - P.sigma) / (1 - P.sigma), 0.0)
    pol = sol['policy']
    part = dict(pol, flow=power(pol['cons']),
                train={key: dict(tr, flow=np.where(tr['weight'] > 0, power(tr['cons']), 0.0)) for key, tr in pol['train'].items()})
    A = np.zeros((P.nh, 2, P.ns, P.na))
    for it in range(max_iter):
        new = apply_transitions(P, sol['T'], policy_value(P, part, A))
        gap = np.max(np.abs(new - A)); A = new
        if gap < tol:
            break
    return A


def solve_values(P, pr, V0=None, tol=1e-9, max_iter=400, sweeps=25):
    """Stationary values: fixed point of start_stage(income_stage(.)).

    Each maximisation is followed by `sweeps` evaluations of the fixed policy.
    """
    V = np.zeros((P.nh, 2, P.ns, P.na)) if V0 is None else V0.copy()
    for it in range(max_iter):
        Vbar, policy = income_stage(P, pr, V)
        Vnew, T, choice = start_stage(P, pr, Vbar)
        gap = np.max(np.abs(Vnew - V))
        V = Vnew
        if gap < tol:
            break
        for _ in range(sweeps):
            V, _, _ = start_stage(P, pr, policy_value(P, policy, V))
    return dict(V=V, Vbar=Vbar, policy=policy, T=T, choice=choice, iterations=it + 1, gap=gap)


def _index(P, k, s):
    return (k * P.ns + s) * P.na


def choice_matrix(P, sol, h):
    """Income stage -> next start of quarter, for one type (dense, rows sum to one)."""
    ns, na, a = P.ns, P.na, P.agrid
    n = 2 * ns * na
    T2 = np.zeros((n, n))
    rows = np.arange(na)
    entry_k = np.array([1 - P.entry_access[h], P.entry_access[h]])

    def add(src, k2, s2, grid_index, prob):
        base = _index(P, k2, s2)
        saving = a[grid_index]
        lo, hi, w = lottery(a, P.estate_survivor * saving)
        np.add.at(T2, (src, base + lo), prob * (1 - P.renewal) * (1 - w))
        np.add.at(T2, (src, base + hi), prob * (1 - P.renewal) * w)

    def add_entrant(src, grid_index, prob):
        saving = a[grid_index]
        lo, hi, w = lottery(a, P.estate_entrant * saving)
        for k2 in range(2):
            for s2 in range(ns):
                p = prob * P.renewal * entry_k[k2] * P.entry_status[h, s2]
                if np.all(p == 0):
                    continue
                base = _index(P, k2, s2)
                np.add.at(T2, (src, base), p * P.estate_zero)
                np.add.at(T2, (src, base + lo), p * (1 - P.estate_zero) * (1 - w))
                np.add.at(T2, (src, base + hi), p * (1 - P.estate_zero) * w)

    for k in range(2):
        for s in range(ns):
            src = _index(P, k, s) + rows
            tr = sol['policy']['train'].get((h, k, s))
            if tr is None:
                j, w = sol['policy']['save'][h, k, s], sol['policy']['mix'][h, k, s]
                for node, share in ((j, 1 - w), (np.minimum(j + 1, na - 1), w)):
                    add(src, k, s, node, share); add_entrant(src, node, share)
            else:
                for z, wz in enumerate(tr['weight']):
                    pi = 1 - np.exp(-P.gamma[h] * tr['course'][z])
                    j, w = tr['save'][z], tr['mix'][z]
                    for node, share in ((j, 1 - w), (np.minimum(j + 1, na - 1), w)):
                        add(src, k, s, node, wz * share * (1 - pi))
                        add(src, 1, unemp(M), node, wz * share * pi)
                        add_entrant(src, node, wz * share)
    return T2


def stationary_distribution(P, sol):
    """Invariant distribution by type. Returns start-of-quarter and income-stage masses."""
    nh, ns, na = P.nh, P.ns, P.na
    n = 2 * ns * na
    mu = np.zeros((nh, 2, ns, na)); mu_income = np.zeros_like(mu)
    for h in range(nh):
        T2 = choice_matrix(P, sol, h)
        Pm = np.zeros((n, n))
        for k in range(2):
            T = sol['T'][(h, k)]
            for s in range(ns):
                dst = slice(_index(P, k, s), _index(P, k, s) + na)
                for s2 in range(ns):
                    p = T[s, s2]
                    if np.any(p):
                        src = slice(_index(P, k, s2), _index(P, k, s2) + na)
                        Pm[dst] += p[:, None] * T2[src]
        A = Pm.T - np.eye(n)
        A[-1] = 1.0
        b = np.zeros(n); b[-1] = P.mass[h]
        x = np.linalg.solve(A, b)
        x = np.maximum(x, 0.0); x *= P.mass[h] / x.sum()
        mu[h] = x.reshape(2, ns, na)
        for k in range(2):
            mu_income[h, k] = np.einsum('sa,sta->ta', mu[h, k], sol['T'][(h, k)])
    return mu, mu_income


def job_values(P, pr, sol, lam, R_next, tol=1e-11, max_iter=5000):
    """Stationary value of a filled job to the employment account, by worker state."""
    nh, na, a = P.nh, P.na, P.agrid
    flow = np.zeros((nh, 2, 2, na)); keep = np.zeros((nh, 2, 2, na))
    nxt = {}
    for h in range(nh):
        stay = 1 - P.exit_E[h].sum()
        for k in range(2):
            for g in range(2):
                if not accessible(P, h, k, g):
                    continue
                flow[h, k, g] = (lam[g] - pr.wage[g]) * P.eff[h, g]
                take = sol['choice'][(h, k)].get(('take', g), 0.0)
                keep[h, k, g] = stay * (1 - take) * (1 - P.separation[g])
                nxt[(h, k, g)] = (sol['policy']['save'][h, k, emp(g)], sol['policy']['mix'][h, k, emp(g)])
    lo, hi, w = lottery(a, P.estate_survivor * a)            # where saving at each node is carried
    J = flow.copy()
    for it in range(max_iter):
        Jp = np.maximum(J, 0.0) * keep
        new = flow.copy()
        for (h, k, g), (j, share) in nxt.items():
            carried = (1 - w) * Jp[h, k, g][lo] + w * Jp[h, k, g][hi]
            new[h, k, g] += (1 - P.renewal) / R_next * between(carried, j, share)
        gap = np.max(np.abs(new - J)); J = new
        if gap < tol:
            break
    return J


def keep_probability(P, choice):
    """Probability that a match continues through the start of a quarter, by worker state.

    Excludes boundary exit, accepted offers from the other occupation and background
    separation. Layoffs are not included: they are the option to dissolve the match.
    """
    keep = np.zeros((P.nh, 2, 2, P.na))
    for h in range(P.nh):
        stay = 1 - P.exit_E[h].sum()
        for k in range(2):
            for g in range(2):
                if accessible(P, h, k, g):
                    keep[h, k, g] = stay * (1 - choice[(h, k)].get(('take', g), 0.0)) * (1 - P.separation[g])
    return keep


def job_value_step(P, wage, lam, policy, R_next, continuation):
    """Value of a filled job before the option to dissolve it, at one date.

    `continuation` is max(J, 0) times the keep probability at the next date.
    """
    a = P.agrid
    J = np.zeros((P.nh, 2, 2, P.na))
    lo, hi, w = lottery(a, P.estate_survivor * a)            # where saving at each node is carried
    for h in range(P.nh):
        for k in range(2):
            for g in range(2):
                if not accessible(P, h, k, g):
                    continue
                c = continuation[h, k, g]
                carried = (1 - w) * c[lo] + w * c[hi]
                J[h, k, g] = ((lam[g] - wage[g]) * P.eff[h, g] + (1 - P.renewal) / R_next
                              * between(carried, policy['save'][h, k, emp(g)], policy['mix'][h, k, emp(g)]))
    return J


def income_distribution(P, mu, T):
    """Start-of-quarter masses -> income-stage masses."""
    out = np.zeros_like(mu)
    for h in range(P.nh):
        for k in range(2):
            out[h, k] = np.einsum('sa,sta->ta', mu[h, k], T[(h, k)])
    return out


def advance(P, policy, mu_income, entrants=True):
    """Income-stage masses -> next start-of-quarter masses.

    With entrants=False only survivors are carried, which follows a fixed cohort.
    """
    nh, ns, na, a = P.nh, P.ns, P.na, P.agrid
    loS, hiS, wS = lottery(a, P.estate_survivor * a)
    loB, hiB, wB = lottery(a, P.estate_entrant * a)

    def spread(mass, lo, hi, w):
        return np.bincount(lo, weights=mass * (1 - w), minlength=na) + np.bincount(hi, weights=mass * w, minlength=na)

    def by_node(j, share, mass):                             # mass at each saving node
        return (np.bincount(j, weights=mass * (1 - share), minlength=na)
                + np.bincount(np.minimum(j + 1, na - 1), weights=mass * share, minlength=na))

    out = np.zeros_like(mu_income)
    for h in range(nh):
        saved = np.zeros(na)
        for k in range(2):
            for s in range(ns):
                m = mu_income[h, k, s]
                if not m.any():
                    continue
                tr = policy['train'].get((h, k, s))
                if tr is None:
                    stay = by_node(policy['save'][h, k, s], policy['mix'][h, k, s], m)
                else:
                    stay = np.zeros(na); qual = np.zeros(na)
                    for z, wz in enumerate(tr['weight']):
                        pi = 1 - np.exp(-P.gamma[h] * tr['course'][z])
                        stay += by_node(tr['save'][z], tr['mix'][z], wz * m * (1 - pi))
                        qual += by_node(tr['save'][z], tr['mix'][z], wz * m * pi)
                    out[h, 1, unemp(M)] += (1 - P.renewal) * spread(qual, loS, hiS, wS)
                    saved += qual
                out[h, k, s] += (1 - P.renewal) * spread(stay, loS, hiS, wS)
                saved += stay
        if entrants:
            born = (1 - P.estate_zero) * spread(saved, loB, hiB, wB)
            born[0] += P.estate_zero * saved.sum()
            for k2, share in enumerate((1 - P.entry_access[h], P.entry_access[h])):
                out[h, k2] += P.renewal * share * P.entry_status[h][:, None] * born[None, :]
    return out


def aggregates(P, pr, sol, mu, mu_income, J=None):
    """Stocks, flows and the household budget identity."""
    nh, ns, na, a = P.nh, P.ns, P.na, P.agrid
    out = {}
    stock = mu_income.sum((0, 1, 3))
    out['employment'] = np.array([stock[emp(g)] for g in range(2)])
    out['unemployment'] = np.array([stock[unemp(g)] for g in range(2)])
    out['outside'] = stock[PRIVATE:]
    out['labour'] = np.array([sum(mu_income[h, :, emp(g)].sum() * P.eff[h, g] for h in range(nh)) for g in range(2)])
    out['payroll'] = float(pr.wage @ out['labour'])
    out['assets_open'] = float((mu.sum((0, 1, 2)) * a).sum())
    out['receipts'] = np.array([(P.receipts[c][:, None, :, None] * mu_income).sum() for c in range(3)])
    out['claims'] = float((P.claim[:, None, None, None] * mu_income).sum())
    out['transfer'] = float(pr.transfer * mu_income.sum())
    cons = (sol['policy']['cons'] * mu_income).sum()
    saving = 0.0; course = 0.0; fees = 0.0; allowance = 0.0; qualified = 0.0; participants = 0.0
    for h in range(nh):
        for k in range(2):
            for s in range(ns):
                m = mu_income[h, k, s]
                tr = sol['policy']['train'].get((h, k, s))
                if tr is None:
                    saving += (m * between(a, sol['policy']['save'][h, k, s], sol['policy']['mix'][h, k, s])).sum()
                else:
                    allowance += P.allowance * m.sum()
                    for z, wz in enumerate(tr['weight']):
                        saving += (wz * m * between(a, tr['save'][z], tr['mix'][z])).sum()
                        course += (wz * m * tr['course'][z]).sum()
                        participants += (wz * m * (tr['course'][z] > 0)).sum()
                        qualified += (wz * m * (1 - np.exp(-P.gamma[h] * tr['course'][z]))).sum()
    out.update(consumption=float(cons), saving=float(saving), instruction=float(course),
               participants=float(participants), qualified=float(qualified), allowance=float(allowance),
               tuition_paid=float((1 - P.subsidy) * P.tuition * course),
               tuition_subsidy=float(P.subsidy * P.tuition * course))
    uses = out['consumption'] + out['tuition_paid'] + out['saving']
    sources = (pr.R * out['assets_open'] + out['payroll'] + out['claims'] * pr.D + out['receipts'].sum() + out['allowance']
               + out['transfer'])
    out['budget_gap'] = float(uses - sources)
    # next-period assets implied by the estate rule (should equal saving)
    out['access_share'] = np.array([mu_income[P.home == g, 1].sum() / mu_income[P.home == g].sum() for g in range(2)])
    # gross flows between statuses within the quarter (start of quarter -> income stage)
    flow = np.zeros((ns, ns))
    for h in range(nh):
        for k in range(2):
            flow += np.einsum('sa,sta->st', mu[h, k], sol['T'][(h, k)])
    out['status_flows'] = flow
    out['status_open'] = mu.sum((0, 1, 3))
    out['zero_assets'] = float(mu_income[..., 0].sum() / mu_income.sum())
    if J is not None:
        out.update(applications(P, pr, sol, mu, J))
    return out


def applications(P, pr, sol, mu, J):
    """Application pools and the average value of a job per application, by occupation."""
    nh, na = P.nh, P.na
    apps = np.zeros(2); value = np.zeros(2); matches = np.zeros(2)
    scale = (pr.finding / P.finding_ref) ** P.entry_elasticity
    for h in range(nh):
        home = P.home[h]
        stay_E, stay_U = 1 - P.exit_E[h].sum(), 1 - P.exit_U[h].sum()
        for k in range(2):
            for g in range(2):
                if not accessible(P, h, k, g):
                    continue
                # searchers attached to g: inherited unemployed plus workers who just lost a job
                ch = sol['choice'][(h, k)]
                take = ch.get(('take', g), 0.0)
                sep = ch.get(('sep', g), P.separation[g])          # background separation and layoffs
                searchers = stay_U * mu[h, k, unemp(g)] + stay_E * (1 - take) * sep * mu[h, k, emp(g)]
                if k == 1:
                    o = 1 - g
                    p = ch[('redirect', g)]
                    for primary, mass in ((g, searchers * (1 - p)), (o, searchers * p)):
                        second = 1 - primary
                        n2 = P.search_intensity[primary] * mass
                        acc2, take2 = ch[('second_accept', g, primary)], ch[('second_take', g, primary)]
                        apps[second] += n2.sum(); value[second] += (n2 * acc2 * np.maximum(J[h, k, second], 0)).sum()
                        matches[second] += pr.finding[second] * (n2 * acc2).sum()
                        n1 = mass * (1 - take2)
                        apps[primary] += n1.sum(); value[primary] += (n1 * np.maximum(J[h, k, primary], 0)).sum()
                        matches[primary] += pr.finding[primary] * n1.sum()
                    # on-the-job applications to the other occupation
                    accept = ch[('accept', g)]
                    mass = P.offer_intensity[g] * stay_E * mu[h, k, emp(g)]
                    apps[o] += mass.sum(); value[o] += (mass * accept * np.maximum(J[h, k, o], 0)).sum()
                    matches[o] += pr.finding[o] * (mass * accept).sum()
                else:
                    apps[g] += searchers.sum(); value[g] += (searchers * np.maximum(J[h, k, g], 0)).sum()
                    matches[g] += pr.finding[g] * searchers.sum()
            for o in range(P.n_out):
                base = P.entry_E[h, o].copy()
                if k == 0:
                    base = np.where(np.arange(2) == home, base.sum(), 0.0)
                mass = mu[h, k, PRIVATE + o]
                for g in range(2):
                    if base[g] > 0 and pr.finding[g] > 0:
                        n = base[g] * scale[g] / pr.finding[g] * mass
                        apps[g] += n.sum(); value[g] += (n * np.maximum(J[h, k, g], 0)).sum()
                        matches[g] += pr.finding[g] * n.sum()
    return dict(applications=apps, job_value_per_application=value / np.maximum(apps, 1e-300), matches=matches)
