"""Production, AI capability and investment (Section 3 of the paper).

The ordinary good combines conventional capital with a CES bundle of knowledge
and manual tasks. AI capital performs knowledge tasks up to the capability
frontier. AI capital is a separate good made with conventional capital. A share
alpha_A of its price pays for that capital; the rest is a rent to its owners.
Two supply rules: 'licence' (the price is a fixed markup over a constant unit
cost, so supply is elastic) and 'scarce' (a scarce input in fixed supply earns
the rent, so the price rises with the amount installed). Z_A = 0 gives the
earlier case in which AI capital is made one for one from the ordinary good.
"""
from dataclasses import dataclass, replace
import math
import numpy as np


@dataclass(frozen=True)
class Technology:
    alpha: float            # cost share of conventional capital
    variety_elasticity: float
    eta: float              # elasticity of substitution between knowledge and manual tasks
    omega: float            # CES weight on knowledge tasks
    Z: float                # total factor productivity
    psi: float              # AI efficiency
    tau: float              # difficulty gradient of AI across tasks
    frontier: float         # share of knowledge tasks AI can perform
    dep_C: float            # depreciation, conventional capital
    dep_A: float            # depreciation, AI capital
    install_C: float        # installation-cost coefficient, conventional
    install_A: float        # installation-cost coefficient, AI
    install_base: float     # K-bar in the installation cost
    alpha_A: float = 1.0    # share of the AI-good price that pays for conventional capital (rest: rent)
    Z_A: float = 0.0        # productivity of the AI-good technology; 0 = AI capital made from the ordinary good
    ai_exponent: float = None   # capital needed is (output/Z_A)**exponent; None = 1/alpha_A (scarce input), 1 = licence
    smooth_C: float = 0.0   # cost of changing the pace of investment, conventional capital
    smooth_A: float = 0.0   # the same for AI capital

    @property
    def nu_A(self):
        return 1 / self.alpha_A if self.ai_exponent is None else self.ai_exponent

    @property
    def marginal_cost(self):
        return (self.variety_elasticity - 1) / self.variety_elasticity

    @property
    def ai_sector(self):
        return self.Z_A > 0


def ai_good(tech, IA, rC):
    """Conventional capital used and the price of the AI good when IA units are produced."""
    if not tech.ai_sector:
        return 0.0 * IA, 1.0 + 0.0 * IA
    IA = np.maximum(IA, 1e-14)
    KCA = (IA / tech.Z_A) ** tech.nu_A
    return KCA, rC * KCA / (tech.alpha_A * IA)


def user_cost(R, dep, install, base, K, price=1.0):
    """Stationary rental that makes replacement investment optimal."""
    I = dep * K
    Q = price + install * I / (K + base)
    return Q * (R - 1 + dep) - install * I * I / (2 * (K + base) ** 2)


def installation(install, base, K, dep):
    I = dep * K
    return install * I * I / (2 * (K + base))


def cutoff(tech, lamN, rA):
    """Share of knowledge tasks performed by AI."""
    if tech.frontier <= 0:
        return 0.0
    ratio = lamN * tech.psi / rA
    if ratio <= 1:
        return 0.0
    return min(tech.frontier, math.log(ratio) / tech.tau)


def knowledge_cost(tech, lamN, rA, m):
    return lamN ** (1 - m) * (rA / tech.psi) ** m * math.exp(tech.tau * m * m / 2)


def bundle_cost(tech, cN, lamM):
    if abs(tech.eta - 1) < 1e-12:                           # the limit of the expression below, the cost dual to `quantities`
        return cN ** tech.omega * lamM ** (1 - tech.omega)   # (corrected on 17 November 2026; this case had not been used)
    e = 1 - tech.eta
    return (tech.omega * cN ** e + (1 - tech.omega) * lamM ** e) ** (1 / e)


def knowledge_share(tech, cN, lamM):
    if abs(tech.eta - 1) < 1e-12:
        return tech.omega
    e = 1 - tech.eta
    return tech.omega * cN ** e / (tech.omega * cN ** e + (1 - tech.omega) * lamM ** e)


def unit_cost(tech, rC, cX):
    return (rC / tech.alpha) ** tech.alpha * (cX / (1 - tech.alpha)) ** (1 - tech.alpha) / tech.Z


@dataclass
class Allocation:
    m: float
    cN: float
    cX: float
    share: float
    rC: float
    rA: float
    Y: float
    Omega: float
    KC: float               # conventional capital in both uses
    KA: float
    labour_demand_N: float
    labour_demand_M: float
    price_gap: float        # log unit cost minus log marginal cost (zero in equilibrium)
    installation: float
    investment: float       # ordinary goods used for investment
    pA: float = 1.0         # price of the AI good
    KCA: float = 0.0        # conventional capital used to make the AI good
    ai_value: float = 0.0   # value of AI-good output
    scarce_rent: float = 0.0


def allocate(tech, R, lamN, lamM, LM, KC_guess=1.0, KA_guess=0.1):
    """Stationary allocation implied by factor prices and manual labour.

    Scale is pinned by manual labour: bundle expenditure Omega = lamM*LM/(1-share).
    Returns knowledge labour demand and the pricing gap; the solver chooses the
    two service prices so that the gap is zero and knowledge labour clears.
    The AI stock solves demand = stock by bisection, because its rental rises
    with the stock when the AI good needs the scarce input.
    """
    KC = KC_guess

    def ai_side(KA, rC):
        KCA, pA = ai_good(tech, tech.dep_A * KA, rC)
        rA = user_cost(R, tech.dep_A, tech.install_A, tech.install_base, KA, pA)
        m = cutoff(tech, lamN, rA)
        cN = knowledge_cost(tech, lamN, rA, m)
        share = knowledge_share(tech, cN, lamM)
        Omega = lamM * LM / (1 - share)
        return m * share * Omega / rA, (KCA, pA, rA, m, cN, share, Omega)

    for _ in range(100):
        rC = user_cost(R, tech.dep_C, tech.install_C, tech.install_base, KC)
        lo, hi = 1e-12, 1.0
        while ai_side(hi, rC)[0] > hi and hi < 1e12:
            hi *= 4
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if ai_side(mid, rC)[0] > mid else (lo, mid)
            if hi - lo < 1e-15 * max(1.0, hi):
                break
        KA = 0.5 * (lo + hi)
        KCA, pA, rA, m, cN, share, Omega = ai_side(KA, rC)[1]
        Y = Omega / ((1 - tech.alpha) * tech.marginal_cost)
        KC_new = tech.alpha * tech.marginal_cost * Y / rC + float(KCA)
        if abs(KC_new - KC) < 1e-13 * max(1, KC):
            KC = KC_new
            break
        KC = KC_new
    KCA, pA = float(KCA), float(pA)
    cX = bundle_cost(tech, cN, lamM)
    LN = (1 - m) * share * Omega / lamN
    gap = math.log(unit_cost(tech, rC, cX)) - math.log(tech.marginal_cost)
    Phi = (installation(tech.install_C, tech.install_base, KC, tech.dep_C)
           + installation(tech.install_A, tech.install_base, KA, tech.dep_A))
    ai_value = pA * tech.dep_A * KA if tech.ai_sector else 0.0
    goods = tech.dep_C * KC + (0.0 if tech.ai_sector else tech.dep_A * KA)
    return Allocation(m, cN, cX, share, rC, rA, Y, Omega, KC, KA, LN, LM, gap, Phi, goods,
                      pA, KCA, ai_value, (1 - tech.alpha_A) * ai_value)


def calibrate(alpha, variety_elasticity, eta, tau, frontier, ai_discount,
              dep_C, dep_A, install_C, install_A, install_base,
              R, lamN, lamM, LN, LM, scarce_share=None, ai_supply='scarce'):
    """Choose omega, Z, psi and the AI-good productivity so that the reference is an equilibrium.

    Inputs are the reference service prices and the efficiency units supplied by
    households. AI costs (1 - ai_discount) times knowledge labour on the easiest
    task. The automated share equals the reference frontier. With `scarce_share`
    given, AI capital is a separate good whose price is one at the reference and
    in whose price the rent (scarce input or licence fee) has that share.
    """
    alpha_A = 1.0 if scarce_share is None else 1 - scarce_share
    tech = Technology(alpha, variety_elasticity, eta, 0.5, 1.0, 1.0, tau, frontier,
                      dep_C, dep_A, install_C, install_A, install_base, alpha_A, 0.0,
                      1.0 if ai_supply == 'licence' else None)
    m = frontier
    bill_N = lamN * LN / (1 - m)                 # expenditure on the knowledge composite
    share = bill_N / (bill_N + lamM * LM)
    Omega = bill_N + lamM * LM
    Y = Omega / ((1 - alpha) * tech.marginal_cost)
    KC, KA = 1.0, 0.1
    for _ in range(500):
        rC = user_cost(R, dep_C, install_C, install_base, KC)
        rA = user_cost(R, dep_A, install_A, install_base, KA)          # AI good priced at one
        KA_new = m * share * Omega / rA
        KCA = 0.0 if scarce_share is None else alpha_A * dep_A * KA_new / rC
        KC_new = alpha * tech.marginal_cost * Y / rC + KCA
        done = abs(KC_new - KC) < 1e-14 * max(1, KC) and abs(KA_new - KA) < 1e-14 * max(1, KA)
        KC, KA = KC_new, KA_new
        if done:
            break
    psi = rA / ((1 - ai_discount) * lamN)
    Z_A = 0.0 if scarce_share is None else dep_A * KA / KCA ** (1 / tech.nu_A)
    tech = replace(tech, psi=psi, Z_A=Z_A)
    assert cutoff(replace(tech, frontier=1.0), lamN, rA) > frontier, 'frontier must bind at the reference'
    cN = knowledge_cost(tech, lamN, rA, m)
    if abs(eta - 1) < 1e-12:
        omega = share
    else:
        odds = share / (1 - share) * (lamM / cN) ** (1 - eta)
        omega = odds / (1 + odds)
    tech = replace(tech, omega=omega)
    cX = bundle_cost(tech, cN, lamM)
    Z = (rC / alpha) ** alpha * (cX / (1 - alpha)) ** (1 - alpha) / tech.marginal_cost
    tech = replace(tech, Z=Z)
    check = allocate(tech, R, lamN, lamM, LM, KC, KA)
    assert abs(check.price_gap) < 1e-9 and abs(check.labour_demand_N / LN - 1) < 1e-8 and abs(check.pA - 1) < 1e-8, check
    return tech, check


# ---------------------------------------------------------------------------
# Transition paths: capital is predetermined, so prices follow from quantities.
# All functions below take numpy arrays over dates.
# ---------------------------------------------------------------------------

def task_split(tech, frontier, KA, LN):
    """Share of knowledge tasks performed by AI, given the installed stock and labour.

    The frontier binds when AI is still cheaper than labour on the last feasible
    task. Otherwise the split equalises the two costs on the marginal task.
    """
    frontier, KA, LN = np.broadcast_arrays(np.asarray(frontier, float), np.asarray(KA, float), np.asarray(LN, float))

    def gap(m):                                   # log cost of AI relative to labour on task m
        return np.log(m) + tech.tau * m + np.log(LN) - np.log1p(-m) - np.log(KA * tech.psi)

    top = np.clip(frontier, 1e-12, 1 - 1e-12)
    lo, hi = np.full(top.shape, 1e-12), top.copy()
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        up = gap(mid) > 0
        hi = np.where(up, mid, hi); lo = np.where(up, lo, mid)
    return np.where(gap(top) <= 0, top, 0.5 * (lo + hi))


def quantities(tech, frontier, KC, KA, LN, LM, IA=None):
    """Output, service prices and rentals implied by capital stocks and labour.

    With an AI-good technology, `IA` is the AI investment of the date: it takes
    conventional capital away from the ordinary good and sets the AI-good price.
    """
    m = task_split(tech, frontier, KA, LN)
    XN = np.exp(m * np.log(tech.psi * KA / m) - tech.tau * m * m / 2 + (1 - m) * np.log(LN / (1 - m)))
    if abs(tech.eta - 1) < 1e-12:
        X = (XN / tech.omega) ** tech.omega * (LM / (1 - tech.omega)) ** (1 - tech.omega)
        share = np.full(np.shape(X), tech.omega)
    else:
        e = (tech.eta - 1) / tech.eta
        tN = tech.omega ** (1 / tech.eta) * XN ** e
        tM = (1 - tech.omega) ** (1 / tech.eta) * LM ** e
        X = (tN + tM) ** (1 / e)
        share = tN / (tN + tM)
    if tech.ai_sector:
        IA = np.maximum(np.asarray(IA, float), 1e-14)
        KCA = (IA / tech.Z_A) ** tech.nu_A
        KCY = np.where(KC > KCA, KC - KCA, np.nan)
    else:
        KCA, KCY = np.zeros(np.shape(X)), KC
    Y = tech.Z * KCY ** tech.alpha * X ** (1 - tech.alpha)
    Omega = (1 - tech.alpha) * tech.marginal_cost * Y
    rC = tech.alpha * tech.marginal_cost * Y / KCY
    pA = rC * KCA / (tech.alpha_A * IA) if tech.ai_sector else np.ones(np.shape(X))
    return dict(m=m, Y=Y, share=share, lamN=(1 - m) * share * Omega / LN, lamM=(1 - share) * Omega / LM,
                rA=m * share * Omega / KA, rC=rC, pA=pA, KCA=KCA)


def capital_path(tech, frontier, R, LN, LM, K_start, K_end, guess=None, tol=1e-11, verbose=False, policy=None,
                 I_before=None):
    """Capital stocks that satisfy the investment conditions, given labour and returns.

    Dates 0..T-1 are the transition and date T is the terminal stationary state:
    frontier, LN, LM and R have length T+1, and R[t] is the gross return paid at t.
    K_start and K_end are (KC, KA) at dates 0 and T. Returns K of shape (T+1, 2),
    investment and installation cost of shape (T, 2), and the quantities at 0..T.

    With `policy` (keys floor, response, mpc, base, other, charges, u) the goods
    market is solved together with the capital paths. The rate of capacity use u_t
    makes output equal spending at each date. The return paid at the next date is
    max(lower, floor * u_t**response). With response 0 (the main scenario) it is fixed at
    the floor; with a positive response it rises when spending exceeds normal capacity and
    falls, down to `lower` (the floor if not given), when capacity is unused. Capital earns u_t times its normal-capacity rental, so
    unused capacity lowers investment. Changing investment from one date to the next
    costs (smooth_n / 2) * (I_t - I_{t-1})**2 / I_{t-1} in final goods; this is
    included in the installation cost that is returned. `I_before` is investment at
    the date before the path (replacement investment if omitted). `other` is spending on final goods apart from
    investment and capitalists, `charges` the deductions from output in capitalists'
    income apart from the upkeep of capital (both of length T), and `u` a starting
    path. Returns the solved paths under 'R' and 'u'.

    With policy['anchor_speed'] (robustness, off by default) the income at which
    capitalists' consumption is anchored is not fixed: it starts at policy['base'] (a
    number) and moves each date by this share of the gap to their current income. The
    anchor path is then solved with the other paths and returned under 'base';
    policy['base_path'] is an optional starting path. With policy['delayed'] the same path is not an anchor but the
    income to which capitalists respond: it follows their income at the same speed, and they consume
    policy['ref'] plus the share mpc of its excess over policy['ref'], the income of the reference.

    With policy['purchases'] (policy experiment, off by default) the government buys final goods whenever spending
    would fall short of normal capacity, so that capacity use never falls below one; the purchases are financed by
    the levy and so are a charge on capitalists' income. They are returned under 'G', and 'ugen' is the solved
    unknown (capacity use if above one; otherwise one minus the share of capacity that the government buys, in logs).
    """
    T = len(LN) - 1
    dep = np.array([tech.dep_C, tech.dep_A]); kap = np.array([tech.install_C, tech.install_A])
    chi = np.array([tech.smooth_C, tech.smooth_A])
    base = tech.install_base
    K0, KT = np.asarray(K_start, float), np.asarray(K_end, float)
    n = 2 * (T - 1)
    ns = T if policy is not None else 0
    speed = policy.get('anchor_speed') if policy is not None else None
    nb = T - 1 if speed is not None else 0                   # anchor at dates 1..T-1

    def changes(K, I):
        """Investment at the previous date, and its change at each date and at the next one
        (replacement investment before and after the path). Investment must stay positive."""
        before = np.vstack([dep * K0 if I_before is None else I_before, I[:-1]]); after = np.vstack([I[1:], dep * KT])
        before = np.where(before > 0, before, np.nan); I = np.where(I > 0, I, np.nan)
        return before, I - before, after - I

    def stocks(x):
        return np.vstack([K0, np.exp(x[:n].reshape(T - 1, 2)), KT])

    def regime(x):
        if policy is None:
            return np.ones(T), np.asarray(R, float)
        s = x[n:n + ns]
        u = np.exp(np.maximum(s, 0.0)) if policy.get('purchases') else np.exp(s)
        Rp = np.array(R, float)
        # with `lower` below the floor (a variant, not the main scenario) the return also falls when capacity is unused
        Rp[1:] = np.maximum(policy.get('lower', policy['floor']), policy['floor'] * u ** policy['response'])
        return u, Rp

    def purchases(x):
        """Public purchases as a share of capacity (policy experiment): what is needed to keep capacity in use.
        The unknown s_t is the log of capacity use when it is positive; when it is negative, capacity is used at
        its normal rate and the government buys the share -s_t of capacity."""
        if policy is None or not policy.get('purchases'):
            return np.zeros(T)
        return np.maximum(-x[n:n + ns], 0.0)

    def anchor(x):
        if policy is None or not nb:
            return None if policy is None else policy['base']
        return np.r_[policy['base'], policy['base'] * x[n + ns:]]

    def residual(x):
        K = stocks(x); u, Rp = regime(x)
        I = K[1:] - (1 - dep) * K[:-1]                       # dates 0..T-1
        q = quantities(tech, frontier[:T], K[:T, 0], K[:T, 1], LN[:T], LM[:T], I[:, 1])
        price = np.column_stack([np.ones(T), q['pA']])
        before, dI, dI_next = changes(K, I)
        Q = (price + kap * I / (K[:-1] + base) + chi * dI / before
             - chi * (dI_next / I + dI_next ** 2 / (2 * I ** 2)) / Rp[1:, None])
        seen = u ** policy.get('weight', 1.0) if policy is not None else u    # capacity use that investors take into account
        rent = (seen[:, None] * np.column_stack([q['rC'], q['rA']]))[1:]    # dates 1..T-1
        dPhi = -kap * I[1:] ** 2 / (2 * (K[1:T] + base) ** 2)
        euler = (Q[:-1] - (rent - dPhi + (1 - dep) * Q[1:]) / Rp[1:T, None]).ravel()
        if policy is None:
            return euler
        upkeep = dep * K[:T]
        charges = (policy['charges'] + upkeep[:, 0] + (0.0 if tech.ai_sector else upkeep[:, 1])
                   + (kap * upkeep ** 2 / (2 * (K[:T] + base))).sum(1))
        goods = (I[:, 0] + (0.0 if tech.ai_sector else I[:, 1])
                 + (kap * I ** 2 / (2 * (K[:T] + base)) + chi * dI ** 2 / (2 * before)).sum(1))
        Y = u * q['Y']
        G = purchases(x) * q['Y']                           # financed by the levy: spending by the government, a charge on capitalists
        other = policy['other'] + G; charges = charges + G
        if policy.get('feedback', 0.0) > 0:
            # Solver aid, not part of the model: the wage bill implied by this block differs from the one
            # households were given; let household spending and capitalists' income respond to the difference.
            # The correction is zero once the paths are consistent, so it does not change the solution.
            ratio = u[:, None] * np.column_stack([q['lamN'], q['lamM']]) / policy['lam']
            extra = (policy['payroll'] * (ratio - 1)).sum(1)
            other = other + policy['feedback'] * extra; charges = charges + extra
        b = anchor(x)
        if policy.get('delayed'):                        # consumption follows the smoothed income b around the fixed reference income
            plan = policy['ref'] + policy['mpc'] * (b - policy['ref'])
        else:
            plan = b + policy['mpc'] * (Y - charges - b)
        out = np.r_[euler, (goods + other + plan) / Y - 1]                  # spending equals output
        if nb:                                                              # the anchor follows capitalists' income
            out = np.r_[out, (b[1:] - b[:-1] - speed * ((Y - charges)[:-1] - b[:-1])) / policy['base']]
        return out

    def smooth():
        w = np.linspace(0, 1, T + 1)[1:T, None]
        return (np.log(K0) * (1 - w) + np.log(KT) * w).ravel()

    x = smooth() if guess is None else np.log(guess[1:T]).ravel()
    if policy is not None:
        x = np.r_[x, np.log(policy['u'])]
        if nb:
            path = policy.get('base_path')
            x = np.r_[x, np.ones(nb) if path is None else np.asarray(path, float)[1:] / policy['base']]
    r = residual(x)
    if not np.all(np.isfinite(r)):
        x[:n] = smooth(); r = residual(x)
    for it in range(80):
        if np.max(np.abs(r)) < tol:
            break
        Jm = np.zeros((n + ns + nb, n + ns + nb)); h = 1e-6
        for colour in range(5):                             # a stock enters the conditions of five dates
            for kind in range(2):
                cols = np.array([2 * s + kind for s in range(T - 1) if s % 5 == colour])
                if len(cols) == 0:
                    continue
                xp = x.copy(); xp[cols] += h
                d = (residual(xp) - r) / h
                for c in cols:
                    s = c // 2
                    rows = [2 * t + jj for t in range(max(s - 2, 0), min(s + 3, T - 1)) for jj in range(2)]
                    if ns:
                        rows += [n + t for t in (s, s + 1, s + 2) if t < T]
                        rows += [n + ns + t for t in (s, s + 1, s + 2) if t < nb]
                    Jm[rows, c] = d[rows]
        for colour in range(3):
            if ns:
                dates = np.array([t for t in range(T) if t % 3 == colour])
                xp = x.copy(); xp[n + dates] += h
                d = (residual(xp) - r) / h
                for t in dates:
                    rows = [2 * tt + jj for tt in (t - 1, t) if 0 <= tt < T - 1 for jj in range(2)] + [n + t]
                    rows += [n + ns + t] if t < nb else []
                    Jm[rows, n + t] = d[rows]
            if nb:                                              # the anchor at date j+1: spending then, and two anchor steps
                cols = np.array([j for j in range(nb) if j % 3 == colour])
                xp = x.copy(); xp[n + ns + cols] += h
                d = (residual(xp) - r) / h
                for j in cols:
                    rows = [n + j + 1] + [n + ns + t for t in (j, j + 1) if t < nb]
                    Jm[rows, n + ns + j] = d[rows]
        try:
            dx = np.linalg.solve(Jm, -r)
        except np.linalg.LinAlgError:
            dx = np.linalg.lstsq(Jm, -r, rcond=1e-12)[0]
        dx *= min(1.0, 0.5 / max(np.max(np.abs(dx)), 1e-300))    # no stock moves by more than about 50% in a step
        scale, norm = 1.0, np.linalg.norm(r)
        for _ in range(30):
            xn = x + scale * dx
            rn = residual(xn)
            if np.all(np.isfinite(rn)) and np.linalg.norm(rn) < norm:
                break
            scale *= 0.5
        x, r = xn, rn
        if verbose:
            print(f"    capital path: step {it} scale {scale:.3g} residual {np.max(np.abs(r)):.2e}")
    K = stocks(x); u, Rp = regime(x)
    I = K[1:] - (1 - dep) * K[:-1]
    before, dI, _ = changes(K, I)
    Phi = kap * I ** 2 / (2 * (K[:-1] + base)) + chi * dI ** 2 / (2 * before)     # installation and adjustment
    q = quantities(tech, frontier, K[:, 0], K[:, 1], LN, LM, np.r_[I[:, 1], tech.dep_A * K[-1, 1]])
    goods = I[:, 0] + (0.0 if tech.ai_sector else I[:, 1])
    return dict(K=K, I=I, Phi=Phi, q=q, goods_investment=goods, residual=float(np.max(np.abs(r))), R=Rp, u=u,
                iterations=it, base=anchor(x), G=purchases(x) * q['Y'][:T],
                ugen=np.exp(x[n:n + ns]) if policy is not None else u)


def stationary_capital(tech, frontier, R, LN, LM, guess, u=1.0):
    """Capital stocks at which rentals, at the rate of capacity use u, equal user costs, given labour
    (used as the end of a path)."""
    dep, kap, base = (tech.dep_C, tech.dep_A), (tech.install_C, tech.install_A), tech.install_base

    def gap(x):
        KC, KA = math.exp(x[0]), math.exp(x[1])
        q = quantities(tech, frontier, KC, KA, LN, LM, tech.dep_A * KA)
        return np.array([math.log(u * float(q['rC']) / user_cost(R, dep[0], kap[0], base, KC)),
                         math.log(u * float(q['rA']) / user_cost(R, dep[1], kap[1], base, KA, float(q['pA'])))])

    x = np.log(np.asarray(guess, float)); r = gap(x)
    for _ in range(200):
        if np.max(np.abs(r)) < 1e-11:
            break
        Jm = np.column_stack([(gap(x + 1e-6 * np.eye(2)[j]) - r) / 1e-6 for j in range(2)])
        try:
            dx = np.clip(np.linalg.solve(Jm, -r), -0.5, 0.5)
        except np.linalg.LinAlgError:
            dx = 0.3 * r
        scale = 1.0
        for _ in range(20):
            rn = gap(x + scale * dx)
            if np.all(np.isfinite(rn)) and np.linalg.norm(rn) < np.linalg.norm(r):
                break
            scale *= 0.5
        else:
            dx, scale = 0.3 * r, 1.0; rn = gap(x + dx)       # rentals above user cost: raise the stock
        x, r = x + scale * dx, rn
    return np.exp(x)
