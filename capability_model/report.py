"""Tables, figures and numbers for the manuscript, from saved results.

    python -m capability_model.report

Reads results/ and writes output/tables and output/figures. In the main scenario the return on claims is fixed; it is written by
`run transition` without options. Its variants add `--open-access` (tag _access), `--capitalist-mpc 0.75`
(_mpc75), `--rent-share 0.36` (_rent36), `--payout-share 0.30` (_own30) or `--wage-speed 0.10` (_wage10).
The second state of demand, in which the anchor of capitalists' consumption follows their income, is
written by `--anchor-speed 0.05 --memory 30 --tolerance 3e-4 --tag _anchor05` (and with `--open-access
--tag _access_anchor05`). The variant in which the return follows the use of capacity and can fall to
zero is written by `--rate-lower 0 --step-horizon 240 --quarters 264 --step-size 0.03 --tag _cut_main`.
The control with easier access and constant capability is written by `--open-access --no-capability
--passes 300 --tag _access_noai`. All outcomes are reported up to forty years after the first rise in
capability. The module
writes output/tables/*.tex and output/figures/*.pdf.
No model is solved here, except the reference equilibrium if it is not on disk.
"""
import numpy as np
import json
from . import parameters as par, run, transition as tr, households as hh, production as prod, welfare as wf
from .households import N, M, PRIVATE, emp, unemp
from .wording import plain          # the vocabulary of the paper; applied to every table and to the texts of the figures

OUT = par.ROOT / 'results'
GEN = par.ROOT / 'output/tables'
FIG = par.ROOT / 'output/figures'
DATES = [(1, 4), (2, 8), (4, 16), (6, 24), (10, 40), (20, 80), (40, 159)]       # (years, quarter index)
LONG = 159                                                  # the long run: forty years after the first rise in capability
FIRST = 80                                                  # quarters over which paths are averaged: the first twenty years


def at(x, q):
    """A path at date q, or its average over the first twenty years when q is 'avg'."""
    return x[:FIRST].mean(0) if isinstance(q, str) else x[q]


def present_value(case, x, x0):
    """Percent difference between the present values of the paths x and x0 over the first forty years, discounted at the return."""
    w = case.eco.R ** -np.arange(LONG + 1.0)
    x0 = np.full(LONG + 1, x0) if np.ndim(x0) == 0 else x0[:LONG + 1]
    return 100 * ((w * x[:LONG + 1]).sum() / (w * x0).sum() - 1)


def consumption_loss(case, idx=None):
    """Present value of the consumption of the surviving members of the initial knowledge cohort in the groups `idx` (all if None)."""
    idx = list(range(9)) if idx is None else idx
    return present_value(case, case.p['cohort_groups_consumption'][:, idx].sum(1), case.p['cohort_groups_consumption_reference'][:, idx].sum(1))


BLUE, ORANGE, AQUA, GREY, INK = '#2a78d6', '#eb6834', '#1baf7a', '#8a8a86', '#0b0b0b'


class Case:
    """One solved scenario with its reference."""

    def __init__(self, regime, target=0.35, tag='', rent_share=None, payout_share=None, eta=None, discount=None, offer_knowledge=None,
                 knowledge_access=None, rho_knowledge=None):
        self.eco, self.ref, self.const, _ = run.stationary_case(run.MAIN[0] if eta is None else eta, run.MAIN[1] if discount is None else discount,
                                                                regime, rent_share=rent_share, payout_share=payout_share,
                                                                offer_knowledge=offer_knowledge, knowledge_access=knowledge_access,
                                                                rho_knowledge=rho_knowledge)
        self.p = np.load(OUT / f'transition_{regime}_f{round(100 * target)}{tag}.npz')
        self.sc = self.const['persons_per_unit'] / 1e6
        self.acc, self.ag = self.ref['accounts'], self.ref['households']['ag']
        p, acc, ag = self.p, self.acc, self.ag
        self.T = len(p['output'])
        w0 = self.eco.wage_share * self.ref['lam']
        pct = lambda x, base: 100 * (x / base - 1)
        self.s = dict(
            ai=100 * p['automated'], frontier=100 * p['frontier'],
            output=pct(p['output'], acc['output']), capacity=pct(p['capacity'], acc['output']), use=100 * p['utilisation'],
            cons=pct(p['household_consumption'], acc['household_consumption']),
            capcons=pct(p['capitalist_consumption'], acc['capitalist_consumption']),
            capincome=pct(p['capitalist_consumption'] + p['capitalist_saving'], tr.reference_income(self.eco, self.ref)),
            wagebill=pct(p['payroll'], ag['payroll']), payouts=pct(p['D'], self.ref['D']),
            jobs=pct(p['employment'].sum(1), ag['employment'].sum()),
            supplied=pct(p['labour'] @ w0, ag['labour'] @ w0), used=pct(p['utilisation'] * (p['labour'] @ w0), ag['labour'] @ w0),
            anywork=100 * (p['employment'].sum(1) + p['outside'][:, 2]) / self.ref['households']['mu'].sum(),
            wageK=pct(p['wage'][:, N], w0[N]), wageM=pct(p['wage'][:, M], w0[M]),
            jobsK=p['employment'][:, N] * self.sc, jobsM=p['employment'][:, M] * self.sc,
            unemp=100 * p['unemployment'].sum(1) / (p['unemployment'].sum(1) + p['employment'].sum(1)),
            outside=p['outside'].sum(1) * self.sc,
            inactive=p['outside'][:, 0] * self.sc, retired=p['outside'][:, 1] * self.sc, otherwork=p['outside'][:, 2] * self.sc,
            findK=p['finding'][:, N], findM=p['finding'][:, M],
            train=p['participants'] * self.sc * 1e3, qualified=100 * p['knowledge_home_access'],
            ret=100 * (p['R'] ** 4 - 1),
            # licence rent in the price of AI capital: licensors price at the full-capacity rental and pay u times it
            rentshare=100 * (1 - p['utilisation'] * self.eco.tech.alpha_A),
            machines=pct(p['capital'][:, 0], self.ref['allocation'].KC), aicap=p['capital'][:, 1] / self.ref['allocation'].KA,
            invest=p['investment'][:, 0],
            # policy experiments: the transfer to residents (percent of household consumption in the reference) and
            # public purchases (percent of output in the reference); zero in the other runs
            transfer=100 * (p['transfer'] if 'transfer' in p.files else np.zeros(self.T)) / acc['household_consumption'],
            purchases=100 * (p['purchases'] if 'purchases' in p.files else np.zeros(self.T)) / acc['output'],
            purshare=100 * (p['purchases'] if 'purchases' in p.files else np.zeros(self.T)) / p['output'])
        self.start = dict(
            jobsK=ag['employment'][N] * self.sc, jobsM=ag['employment'][M] * self.sc,
            unemp=100 * ag['unemployment'].sum() / (ag['unemployment'].sum() + ag['employment'].sum()),
            outside=ag['outside'].sum() * self.sc, inactive=ag['outside'][0] * self.sc, retired=ag['outside'][1] * self.sc,
            otherwork=ag['outside'][2] * self.sc, findK=self.ref['finding'][N], findM=self.ref['finding'][M],
            train=ag['participants'] * self.sc * 1e3, qualified=100 * ag['access_share'][N], ai=100 * self.eco.tech.frontier,
            anywork=100 * (ag['employment'].sum() + ag['outside'][2]) / self.ref['households']['mu'].sum())
        self.cohort = self.shares(p['cohort']); self.cohort_ref = self.shares(p['cohort_reference'])
        self.cohort6 = self.statuses(p['cohort']); self.cohort6_ref = self.statuses(p['cohort_reference'])
        # the outside statuses of the cohort: inactive, retired, other employment (percent of surviving members)
        split = lambda c: 100 * c[:, PRIVATE:] / c.sum(1)[:, None]
        self.cohort_out = split(p['cohort']) - split(p['cohort_reference'])
        # welfare: the values of residents at the initial date as they turn out, by state (written by runs since 17 November 2026)
        self.W = None
        if 'welfare_start' in p.files:
            W = np.array(p['welfare_start'])
            row = json.load(open(OUT / f'transition{tag}.json'))['rows'][0]
            moved = lambda x: x                         # residents who are qualified at once: the value of the state they are moved to
            if row.get('open_access'):
                def moved(x):
                    x = np.array(x)
                    for h in np.flatnonzero(self.eco.P.home == N):
                        x[..., h, 0, :, :] = x[..., h, 1, :, :]
                    return x
            self.W = moved(W)
            side = OUT / f'welfare{tag}.npz'                # written by welfare.py: other discounting, later dates
            self.more = None
            if side.exists():
                m = np.load(side)
                self.more = dict(start=moved(m['start_social']), own=moved(m['later_own']), social=moved(m['later_social']),
                                 dates=list(m['dates']), entrants=m['entrants'], beta=float(m['beta']), check=float(m['check']))

    def welfare(self, weights):
        """Consumption-equivalent change in the welfare of the residents with the masses `weights` (by state at the
        initial date), in percent: the change of consumption at every date and in every state of the reference that
        gives them, on average, the value that the path gives them."""
        return self.equivalent(weights, self.W)

    def equivalent(self, weights, values, social=False):
        """Consumption-equivalent change (percent) that gives the residents with the masses `weights` the average of
        `values`; with `social` the reference is valued at the return on claims instead of the residents' own discounting."""
        if not hasattr(self, '_parts'):
            self._parts = {}
        if social not in self._parts:
            self._parts[social] = wf.reference_parts(self.eco.P, self.ref['households']['sol'], self.more['beta'] if social else None)
        A, B = self._parts[social]
        on = weights > 0
        gain = (weights[on] * (values[on] - B[on])).sum() / (weights[on] * A[on]).sum()
        return 100 * (gain ** (1 / (1 - self.eco.P.sigma)) - 1)

    def social(self, weights):
        """The same change when every date is discounted at the return on claims (choices unchanged)."""
        return self.equivalent(weights, self.more['start'], social=True)

    def entrant(self, quarter, social=False):
        """Consumption-equivalent change in the welfare of a resident who enters in `quarter`, with the states of the
        residents who enter in a quarter of the reference."""
        i = self.more['dates'].index(quarter)
        return self.equivalent(self.more['entrants'], self.more['social' if social else 'own'][i], social=social)

    def residents(self):
        """Groups of the residents alive at the initial date, as masses by state."""
        mu = self.ref['households']['mu']
        only = lambda statuses: np.where(np.isin(np.arange(self.eco.P.ns), statuses)[None, None, :, None], mu, 0.0)
        groups = tr.start_state(self.eco, self.ref)['groups']
        out = dict(All=mu, Cohort=only([emp(N), unemp(N)]), Manual=only([emp(M), unemp(M)]),
                   Inactive=only([PRIVATE]), Retired=only([PRIVATE + 1]), OtherWork=only([PRIVATE + 2]))
        out.update({name: groups[idx].sum(0) for name, idx in Case.GROUPS.items()})
        knowledge_home = (self.eco.P.home == N)[:, None, None, None]     # the residents in manual work, by home occupation
        out['ManualK'] = np.where(knowledge_home, out['Manual'], 0.0); out['ManualM'] = np.where(knowledge_home, 0.0, out['Manual'])
        return out

    # the groups of the cohort (transition.cohort_groups): class c of qualification and third j of savings are at 3 * c + j
    GROUPS = dict(NoQual=[0, 1, 2], Qual=[3, 4, 5], ManualHome=[6, 7, 8], LowSave=[0, 3, 6], MidSave=[1, 4, 7], HighSave=[2, 5, 8])

    def group(self, idx, q):
        """Members of the initial knowledge cohort in the groups `idx` at date q, against the same members in the reference.

        Shares are percent of surviving members; gaps are percentage points; consumption and assets are percent differences
        per surviving member."""
        p = self.p
        x, x0 = p['cohort_groups'][:, idx].sum(1), p['cohort_groups_reference'][:, idx].sum(1)      # (date, qualified, status)
        def split(y):
            st = y[q].sum(0)
            return 100 * np.array([st[emp(N)], st[emp(M)], st[unemp(N)] + st[unemp(M)] + st[PRIVATE] + st[PRIVATE + 1],
                                   st[PRIVATE + 2]]) / st.sum()
        s, s0 = split(x), split(x0)
        per = lambda key, ref: p[key + ('_reference' if ref else '')][q, idx].sum() / (x0 if ref else x)[q].sum()
        return dict(share=100 * x0[0].sum() / p['cohort_groups_reference'][0].sum(),
                    outK=s0[0] - s[0], intoM=s[1] - s0[1], nowork=s[2] - s0[2], other=s[3] - s0[3],
                    jobM=s[1], jobMref=s0[1], jobK=s[0], jobKref=s0[0],
                    cons=100 * (per('cohort_groups_consumption', False) / per('cohort_groups_consumption', True) - 1),
                    assets=100 * (per('cohort_groups_assets', False) / per('cohort_groups_assets', True) - 1),
                    qualified=100 * x[q, 1].sum() / x[q].sum(), qualifiedref=100 * x0[q, 1].sum() / x0[q].sum())

    def home_manual(self):
        """Percent of manual-home residents who hold a manual job: the path, and the reference."""
        sb = self.p['status_by_home'][:, M]
        mu = self.ref['households']['mu_income'][self.eco.P.home == M].sum((0, 1, 3))
        return 100 * sb[:, emp(M)] / sb.sum(1), 100 * mu[emp(M)] / mu.sum()

    @staticmethod
    def statuses(c):
        """Percent of the surviving initial knowledge cohort: knowledge job, manual job, other employment, unemployed,
        inactive or retired, and without work (the sum of the last two)."""
        total = c.sum(1)
        u = c[:, unemp(N)] + c[:, unemp(M)]; out = c[:, PRIVATE] + c[:, PRIVATE + 1]
        return 100 * np.column_stack([c[:, emp(N)], c[:, emp(M)], c[:, PRIVATE + 2], u, out, u + out]) / total[:, None]

    @staticmethod
    def shares(c):
        """Percent of the surviving initial knowledge cohort: knowledge job, manual job, unemployed, outside, no private job."""
        total = c.sum(1)
        k, m = c[:, emp(N)], c[:, emp(M)]
        u = c[:, unemp(N)] + c[:, unemp(M)]; o = c[:, PRIVATE:].sum(1)
        return 100 * np.column_stack([k, m, u, o, u + o]) / total[:, None]


def boom(case):
    """The peak of output in a case and the spending behind it, in points of reference output (Appendix D)."""
    p, acc, v = case.p, case.acc, case.s
    q = int(v['output'][:LONG + 1].argmax())
    parts = dict(invest=p['investment'][q, 0] - acc['investment'], capcons=p['capitalist_consumption'][q] - acc['capitalist_consumption'],
                 households=p['household_consumption'][q] - acc['household_consumption'])
    parts = {k: 100 * x / acc['output'] for k, x in parts.items()}
    parts['other'] = v['output'][q] - sum(parts.values())
    gap = case.cohort[:LONG + 1, 2] - case.cohort_ref[:LONG + 1, 2] + case.cohort_out[:LONG + 1, 0] + case.cohort_out[:LONG + 1, 1]
    return dict(parts, quarter=q, peak=v['output'][q], year=q // 4 + 1, slack=int(np.argmax(v['use'] < 100)) // 4 + 1,
                zero=int(np.argmax(v['ret'] < 0.01)) // 4 + 1, retmax=v['ret'].max(), usemax=v['use'].max(),
                nowork=gap[:FIRST].mean(), worst=gap.max(), worstyear=int(gap.argmax()) // 4 + 1,
                output=v['output'][LONG], cons=v['cons'][LONG])


def f(x, d=1, sign=False):
    x = float(x)
    if d < 0:                                                # round to tens, hundreds, ...
        x, d = round(x, d), 0
    if round(abs(x), d) == 0:
        x = 0.0
    s = f"{x:+,.{d}f}" if sign else f"{x:,.{d}f}"
    return s.replace('-', '$-$') if '$' not in s else s


def numbers(main, pol, mpc, rent, anc, own, slow, acc, ancacc, ctl, tra, pur, tests, checks):
    """Macros used in the text. `main` is the experiment (the return on claims is fixed); `anc` is the second state of
    demand, in which the anchor of capitalists' consumption follows their income; `pol` is the variant in which the
    return follows the use of capacity and can fall to zero."""
    m, q6, qL = main.s, 24, LONG
    out = {}
    def put(name, value, d=0):
        out[name] = f(value, d)
    put('AIStart', main.start['ai']); put('AITarget', m['frontier'][qL]); put('AIShareSix', m['ai'][q6]); put('AIShareLong', m['ai'][qL])
    for tag, q in (('Two', 8), ('Four', 16), ('Six', q6), ('Ten', 40), ('Long', qL)):
        put('OutputFall' + tag, -m['output'][q]); put('ConsFall' + tag, -m['cons'][q])
        put('KWageFall' + tag, -m['wageK'][q]); put('MWageFall' + tag, -m['wageM'][q]); put('WageBillFall' + tag, -m['wagebill'][q])
        put('KJobs' + tag, m['jobsK'][q], 1); put('MJobs' + tag, m['jobsM'][q], 1)
        put('KJobsFallPct' + tag, -100 * (m['jobsK'][q] / main.start['jobsK'] - 1)); put('MJobsRisePct' + tag, 100 * (m['jobsM'][q] / main.start['jobsM'] - 1))
        put('Unemp' + tag, m['unemp'][q], 1); put('Outside' + tag, m['outside'][q], 1); put('OutsideRise' + tag, m['outside'][q] - main.start['outside'], 1)
        put('Use' + tag, m['use'][q]); put('Qual' + tag, m['qualified'][q], 1)
        put('CohK' + tag, main.cohort[q, 0], 1); put('CohKRef' + tag, main.cohort_ref[q, 0], 1)
        put('CohM' + tag, main.cohort[q, 1], 1); put('CohMRef' + tag, main.cohort_ref[q, 1], 1)
        put('CohNoJob' + tag, main.cohort[q, 4], 1); put('CohNoJobRef' + tag, main.cohort_ref[q, 4], 1)
        put('CohNoJobGap' + tag, main.cohort[q, 4] - main.cohort_ref[q, 4], 1)
        put('CohNoWork' + tag, main.cohort6[q, 5], 1); put('CohNoWorkRef' + tag, main.cohort6_ref[q, 5], 1)
        put('CohKGap' + tag, main.cohort_ref[q, 0] - main.cohort[q, 0], 1); put('CohMGap' + tag, main.cohort[q, 1] - main.cohort_ref[q, 1], 1)
    year = lambda cond: int(np.argmax(cond)) // 4 + 1
    put('MachinesFallSix', -m['machines'][q6]); put('MachinesFallLong', -m['machines'][qL])
    put('AvgCohOther', float((main.cohort6 - main.cohort6_ref)[:FIRST, 2].mean()), 1)     # other employment, average over years 1-20
    # the same aggregates with one decimal, as in Table 3 (Section 6.2)
    put('OutputFallSixOne', -m['output'][q6], 1); put('OutputFallLongOne', -m['output'][qL], 1)
    put('UseSixOne', m['use'][q6], 1); put('AIShareSixOne', m['ai'][q6], 1)
    put('WageBillFallSixOne', -m['wagebill'][q6], 1); put('ConsFallSixOne', -m['cons'][q6], 1); put('ConsFallLongOne', -m['cons'][qL], 1)
    put('MachinesFallLongOne', -m['machines'][qL], 1)
    put('CapIncomeFallSix', -m['capincome'][q6]); put('UseMin', m['use'].min()); put('UseMax', m['use'].max(), 1)
    put('FindKMinYear', year(m['findK'] == m['findK'].min())); put('UnempPeakYear', year(m['unemp'] == m['unemp'].max()))
    put('ReturnSpread', m['ret'].max() - m['ret'].min(), 3)
    nowork = lambda case, q: case.cohort[q, 2] - case.cohort_ref[q, 2] + case.cohort_out[q, 0] + case.cohort_out[q, 1]
    gaps = np.array([nowork(main, q) for q in range(qL + 1)])
    put('CohNoWorkGapPeak', gaps.max(), 1); put('CohNoWorkGapPeakYear', int(gaps.argmax()) // 4 + 1)
    put('PathYears', main.T // 4)
    # the variant in which the return follows the use of capacity and can fall to zero (robustness)
    c = pol.s
    put('PolOutputRiseSix', c['output'][q6]); put('PolOutputPeak', c['output'].max()); put('PolPeakYear', year(c['output'] == c['output'].max()))
    put('PolUsePeak', c['use'].max()); put('PolUseLong', c['use'][qL]); put('PolReturnMax', c['ret'].max(), 1)
    put('PolSlackYear', year(c['use'] < 100)); put('PolZeroYear', year(c['ret'] < 0.01)); put('PolBelowYear', year(c['output'] < 0))
    put('PolMachinesRiseSix', c['machines'][q6]); put('PolMachinesRiseLong', c['machines'][qL]); put('PolCapacityRiseLong', c['capacity'][qL])
    put('PolCapConsRiseSix', c['capcons'][q6]); put('PolMWageRiseSix', c['wageM'][q6]); put('PolMWageBelowYear', year(c['wageM'] < 0))
    put('PolKWageFallSix', -c['wageK'][q6]); put('PolConsChangeSix', abs(c['cons'][q6]), 1); put('PolWageBillChangeSix', abs(c['wagebill'][q6]), 1)
    put('PolOutputFallLong', -c['output'][qL]); put('PolConsFallLong', -c['cons'][qL]); put('PolWageBillFallLong', -c['wagebill'][qL])
    put('PolKWageFallLong', -c['wageK'][qL]); put('PolMWageFallLong', -c['wageM'][qL])
    put('PolKJobsLong', c['jobsK'][qL], 1); put('PolOutsideRiseLong', c['outside'][qL] - pol.start['outside'], 1)
    put('PolUnempMin', c['unemp'].min(), 1); put('PolUnempMax', c['unemp'].max(), 1)
    gaps = np.array([nowork(pol, q) for q in range(qL + 1)])
    put('PolNoWorkPeak', gaps.max(), 1); put('PolNoWorkPeakYear', int(gaps.argmax()) // 4 + 1)
    put('PolPathYears', pol.T // 4)
    put('PolLongSettled', max(abs(c[k][pol.T - 1] - c[k][qL]) for k in ('output', 'cons', 'wageK', 'wageM')), 1)
    for tag, q in (('Six', q6), ('Ten', 40), ('Long', qL)):
        put('PolCohKGap' + tag, pol.cohort_ref[q, 0] - pol.cohort[q, 0], 1); put('PolCohMGap' + tag, pol.cohort[q, 1] - pol.cohort_ref[q, 1], 1)
    # the same scenario when capitalists consume 0.75 of an increase in their income (robustness)
    s75 = mpc.s
    for tag, q in (('Two', 8), ('Four', 16), ('Six', q6), ('Long', qL)):
        put('MpcOutputFall' + tag, -s75['output'][q]); put('MpcConsFall' + tag, -s75['cons'][q])
    put('MpcCapConsFallSix', -s75['capcons'][q6]); put('CapConsFallSix', -m['capcons'][q6])
    put('MpcKWageFallLong', -s75['wageK'][qL]); put('MpcMWageFallLong', -s75['wageM'][qL])
    put('MpcCohortGap', max(abs(mpc.cohort[q, j] - main.cohort[q, j]) for q in range(main.T) for j in (0, 1, 4)), 1)
    # the same scenario when the licence rent is 36 percent of the price of AI capital (robustness)
    r36 = rent.s
    above = np.r_[np.where(r36['output'][:80] > 0)[0], 0] if not (r36['output'][:80] > 0).any() else np.where(r36['output'][:80] > 0)[0]
    put('RentAboveFirst', above.min() // 4 + 1); put('RentAboveLast', above.max() // 4 + 1); put('RentOutputPeak', r36['output'].max(), 1)
    put('RentUseMax', r36['use'].max())
    put('RentOutputFallSixOne', -r36['output'][q6], 1)                # with one decimal, for Section 8.3
    put('RentReturnMax', r36['ret'].max(), 1)
    for tag, q in (('Six', q6), ('Long', qL)):
        put('RentOutputFall' + tag, -r36['output'][q]); put('RentConsFall' + tag, -r36['cons'][q])
        put('RentKWageFall' + tag, -r36['wageK'][q]); put('RentMWageFall' + tag, -r36['wageM'][q])
        put('RentCohKGap' + tag, rent.cohort_ref[q, 0] - rent.cohort[q, 0], 1); put('RentCohMGap' + tag, rent.cohort[q, 1] - rent.cohort_ref[q, 1], 1)
        put('RentCohNoJobGap' + tag, rent.cohort[q, 4] - rent.cohort_ref[q, 4], 1)
    put('RentKJobsSix', r36['jobsK'][q6], 1)
    put('LicenceShareRef', 100 * (1 - main.eco.tech.alpha_A)); put('RentLicenceShareRef', 100 * (1 - rent.eco.tech.alpha_A))
    put('LicenceShareSix', m['rentshare'][q6]); put('LicenceShareLong', m['rentshare'][qL])
    put('RentLicenceShareSix', r36['rentshare'][q6]); put('RentLicenceShareLong', r36['rentshare'][qL])
    put('RentLicenceShareMin', r36['rentshare'].min())
    # the same scenario when the anchor of capitalists' consumption follows their income (robustness)
    a = anc.s
    above = np.r_[np.where(a['output'][:80] > 0)[0], 0] if not (a['output'][:80] > 0).any() else np.where(a['output'][:80] > 0)[0]
    put('AncAboveFirst', above.min() // 4 + 1); put('AncAboveLast', above.max() // 4 + 1); put('AncOutputPeak', a['output'].max(), 1)
    put('AncOutputFallTwo', -a['output'][8], 1); put('AncOutputFallLong', -a['output'][qL], 1)
    put('AncIncomeRiseSix', a['capincome'][q6]); put('AncIncomeRiseLong', a['capincome'][qL])
    put('AncCapConsRiseSix', a['capcons'][q6]); put('AncCapConsRiseLong', a['capcons'][qL])
    put('AncCapacityRiseLong', a['capacity'][qL]); put('AncUseLong', a['use'][qL]); put('AncMachinesRiseLong', a['machines'][qL])
    put('AncWageBillFallLong', -a['wagebill'][qL]); put('AncOutsideRiseLong', a['outside'][qL] - anc.start['outside'], 1)
    for tag, q in (('Six', q6), ('Long', qL)):
        put('AncConsFall' + tag, -a['cons'][q]); put('AncKWageFall' + tag, -a['wageK'][q]); put('AncMWageFall' + tag, -a['wageM'][q])
        put('AncKJobs' + tag, a['jobsK'][q], 1)
        put('AncCohKGap' + tag, anc.cohort_ref[q, 0] - anc.cohort[q, 0], 1); put('AncCohMGap' + tag, anc.cohort[q, 1] - anc.cohort_ref[q, 1], 1)
        put('AncCohNoJobGap' + tag, anc.cohort[q, 4] - anc.cohort_ref[q, 4], 1)
    # the same scenario when residents hold 30 percent of the claims on business payouts (robustness)
    o = own.s
    put('OwnShare', 100 * own.ag['claims'])
    put('LiquidAssetsRef', main.ag['assets_open'] / main.acc['gdp'], 2); put('OwnLiquidAssets', own.ag['assets_open'] / own.acc['gdp'], 2)
    put('ZeroAssetsRef', 100 * main.ag['zero_assets'], 1); put('OwnZeroAssets', 100 * own.ag['zero_assets'], 1)
    put('OwnPayoutIncomeShare', 100 * own.ag['claims'] * own.ref['D'] / own.acc['household_consumption'], 1)
    put('OwnCapConsRefFall', -100 * (own.acc['capitalist_consumption'] / main.acc['capitalist_consumption'] - 1))
    put('OwnPayoutFallSix', -o['payouts'][q6], 1); put('OwnPayoutFallLong', -o['payouts'][qL], 1)
    put('OwnConsGainLong', o['cons'][qL] - m['cons'][qL], 1); put('OwnIncomeFallSix', -o['capincome'][q6])
    for tag, q in (('Six', q6), ('Long', qL)):
        put('OwnOutputFall' + tag, -o['output'][q]); put('OwnConsFall' + tag, -o['cons'][q])
        put('OwnKWageFall' + tag, -o['wageK'][q]); put('OwnMWageFall' + tag, -o['wageM'][q])
        put('OwnCohKGap' + tag, own.cohort_ref[q, 0] - own.cohort[q, 0], 1); put('OwnCohMGap' + tag, own.cohort[q, 1] - own.cohort_ref[q, 1], 1)
        put('OwnCohNoJobGap' + tag, own.cohort[q, 4] - own.cohort_ref[q, 4], 1)
    # the same scenario when posted wages close 10 percent of the gap each quarter (robustness)
    w = slow.s
    put('SlowLayoffs', slow.p['layoffs'].sum() * slow.sc, 1)
    put('SlowKJobsMin', w['jobsK'].min(), 1); put('SlowKJobsMinYear', int(w['jobsK'].argmin()) // 4 + 1)
    put('SlowKJobsGapSix', m['jobsK'][q6] - w['jobsK'][q6], 1)
    put('SlowFindKMin', w['findK'].min(), 2); put('SlowFindMMin', w['findM'].min(), 2)
    put('SlowUnempPeak', w['unemp'].max(), 1); put('SlowUnempLong', w['unemp'][qL], 1)
    rise = w['outside'] - slow.start['outside']
    put('SlowOutsidePeak', rise.max(), 1); put('SlowOutsidePeakYear', int(rise.argmax()) // 4 + 1)
    over = slow.p['wage'][:, N] / slow.p['lam'][:, N]
    put('SlowWageAboveQuarters', int((over > 1).sum())); put('SlowWageAboveMax', 100 * (over.max() - 1), 1)
    for tag, q in (('Six', q6), ('Long', qL)):
        put('SlowOutputFall' + tag, -w['output'][q]); put('SlowConsFall' + tag, -w['cons'][q])
        put('SlowKWageFall' + tag, -w['wageK'][q]); put('SlowMWageFall' + tag, -w['wageM'][q])
        put('SlowKJobs' + tag, w['jobsK'][q], 1); put('SlowMJobs' + tag, w['jobsM'][q], 1)
        put('SlowOutsideRise' + tag, rise[q], 1)
        put('SlowCohKGap' + tag, slow.cohort_ref[q, 0] - slow.cohort[q, 0], 1); put('SlowCohMGap' + tag, slow.cohort[q, 1] - slow.cohort_ref[q, 1], 1)
        put('SlowCohNoJobGap' + tag, slow.cohort[q, 4] - slow.cohort_ref[q, 4], 1)
    # averages over the first twenty years, so that comparisons do not rest on a single date
    avg = lambda x: float(np.mean(x[:FIRST]))
    every10 = (('', main), ('Pol', pol), ('Tra', tra), ('Pur', pur), ('Anc', anc), ('Acc', acc), ('AncAcc', ancacc),
               ('Mpc', mpc), ('Rent', rent), ('Own', own), ('Slow', slow),
               ('Eta', checks[0]), ('Cost', checks[1]), ('Whole', checks[2]))
    means = {}
    for prefix, case in every10:
        v = case.s
        k = dict(nowork=avg([nowork(case, q) for q in range(FIRST)]), m=avg(case.cohort[:, 1] - case.cohort_ref[:, 1]),
                 k=avg(case.cohort_ref[:, 0] - case.cohort[:, 0]), wageM=avg(v['wageM']), wageK=avg(v['wageK']),
                 output=avg(v['output']), cons=avg(v['cons']), use=avg(v['use']))
        means[prefix] = k
        put(prefix + 'AvgNoWork', k['nowork'], 1); put(prefix + 'AvgCohM', k['m'], 1); put(prefix + 'AvgCohK', k['k'], 1)
        put(prefix + 'AvgManualShare', 100 * k['m'] / k['k']); put(prefix + 'AvgMWageFall', -k['wageM']); put(prefix + 'AvgKWageFall', -k['wageK'])
        put(prefix + 'AvgOutputChange', abs(k['output'])); put(prefix + 'AvgConsFall', -k['cons']); put(prefix + 'AvgUse', k['use'])
        put(prefix + 'AvgOutputFallOne', -k['output'], 1)
        put(prefix + 'AvgMJobsPct', 100 * (avg(v['jobsM']) / case.start['jobsM'] - 1)); put(prefix + 'AvgMJobs', avg(v['jobsM']), 1)
        put(prefix + 'AvgFindM', avg(v['findM']), 2); put(prefix + 'AvgHomeM', avg(case.home_manual()[0]), 1)
    trio = (('Acc', ''), ('AncAcc', 'Anc'))
    for a_, b_ in trio:
        put(a_ + 'AvgMShift', means[a_]['m'] - means[b_]['m'], 1); put(a_ + 'AvgMWageGap', means[b_]['wageM'] - means[a_]['wageM'])
    put('AccAvgMShiftMin', min(means[a_]['m'] - means[b_]['m'] for a_, b_ in trio), 1)
    put('AccAvgMShiftMax', max(means[a_]['m'] - means[b_]['m'] for a_, b_ in trio), 1)
    put('AccAvgNoWorkShiftMax', max(abs(means[a_]['nowork'] - means[b_]['nowork']) for a_, b_ in trio), 1)
    put('AccAvgMWageGapMin', min(means[b_]['wageM'] - means[a_]['wageM'] for a_, b_ in trio))
    put('AccAvgMWageGapMax', max(means[b_]['wageM'] - means[a_]['wageM'] for a_, b_ in trio))
    variants5 = ('', 'Mpc', 'Rent', 'Own', 'Slow')
    put('VarAvgNoWorkMin', min(means[x]['nowork'] for x in variants5), 1); put('VarAvgNoWorkMax', max(means[x]['nowork'] for x in variants5), 1)
    put('AvgYears', FIRST // 4)
    # the profile of the path in the experiment and in its four variants (Section 8)
    for prefix, case in (('', main), ('Mpc', mpc), ('Rent', rent), ('Own', own), ('Slow', slow)):
        v = case.s
        put(prefix + 'BoomPeak', v['output'].max()); put(prefix + 'BoomPeakYear', year(v['output'] == v['output'].max()))
        put(prefix + 'BoomEnd', year(v['use'] < 100)); put(prefix + 'ZeroFrom', year(v['ret'] < 0.01)); put(prefix + 'BelowFrom', year(v['output'] < 0))
        put(prefix + 'RetMax', v['ret'].max(), 1); put(prefix + 'UnempMax', v['unemp'].max(), 1)
        put(prefix + 'ConsPeak', v['cons'].max(), 1); put(prefix + 'MWagePeak', v['wageM'].max()); put(prefix + 'CapConsMax', v['capcons'].max())
        gap = np.array([nowork(case, q) for q in range(qL + 1)])
        put(prefix + 'NoWorkPeak', gap.max(), 1); put(prefix + 'NoWorkPeakYear', int(gap.argmax()) // 4 + 1)
        rise = v['outside'][:qL + 1] - case.start['outside']
        put(prefix + 'OutsideMax', rise.max(), 1); put(prefix + 'OutsideMaxYear', int(rise.argmax()) // 4 + 1)
        put(prefix + 'KJobsLow', v['jobsK'][:qL + 1].min(), 1); put(prefix + 'KJobsLowYear', int(v['jobsK'][:qL + 1].argmin()) // 4 + 1)
        put(prefix + 'FindKLow', v['findK'].min(), 2); put(prefix + 'FindMLow', v['findM'].min(), 2)
        for tag, q in (('Six', q6), ('Ten', 40), ('Long', qL)):
            put(prefix + 'OutputChange' + tag, abs(v['output'][q])); put(prefix + 'ConsChange' + tag, abs(v['cons'][q]))
            put(prefix + 'KWageChange' + tag, abs(v['wageK'][q])); put(prefix + 'MWageAbs' + tag, abs(v['wageM'][q]))
            put(prefix + 'UseAt' + tag, v['use'][q]); put(prefix + 'MachinesChange' + tag, abs(v['machines'][q]))
            put(prefix + 'KJobsAt' + tag, v['jobsK'][q], 1); put(prefix + 'MJobsAt' + tag, v['jobsM'][q], 1)
            put(prefix + 'OutsideChange' + tag, abs(v['outside'][q] - case.start['outside']), 1)
            put(prefix + 'CohKGap' + tag, case.cohort_ref[q, 0] - case.cohort[q, 0], 1); put(prefix + 'CohMGap' + tag, case.cohort[q, 1] - case.cohort_ref[q, 1], 1)
            put(prefix + 'CohNoWorkGap' + tag, nowork(case, q), 1)
    put('MpcCohortGapLong', max(abs(mpc.cohort[qL, j] - main.cohort[qL, j]) for j in (0, 1, 4)), 1)
    put('MpcLongGap', max(abs(mpc.s[k][qL] - m[k][qL]) for k in ('output', 'cons', 'wageK', 'wageM')), 1)
    put('OwnPayoutRiseSix', own.s['payouts'][q6]); put('PayoutRiseSix', m['payouts'][q6]); put('OwnPayoutChangeLong', abs(own.s['payouts'][qL]), 1)
    put('PayoutChangeLong', abs(m['payouts'][qL]), 1); put('OwnConsRiseSix', own.s['cons'][q6], 1)
    put('SlowLayoffsMax', slow.p['layoffs'].max() * slow.sc, 1)
    # ranges over the experiment and the six robustness checks, forty years after the first rise in capability
    every = (main, mpc, rent, anc, own, slow)
    span = lambda name, values: (put('Rob' + name + 'Min', min(values)), put('Rob' + name + 'Max', max(values)))
    span('ConsFall', [-c.s['cons'][qL] for c in every])
    span('WageFall', [-c.s[k][qL] for c in every for k in ('wageK', 'wageM')])
    span('WageBillFall', [-c.s['wagebill'][qL] for c in every])
    span('KJobsFall', [-100 * (c.s['jobsK'][qL] / c.start['jobsK'] - 1) for c in every])
    span('OutputFallLong', [-c.s['output'][qL] for c in every])
    put('RobUnempPeakMax', max(c.s['unemp'].max() for c in every), 1)
    put('RobLayoffs', sum(c.p['layoffs'].sum() * c.sc for c in every), 1)
    put('RobAboveCount', sum(bool(c.s['output'].max() > 0.5) for c in every))
    put('SlowKWageChangeTenSigned', slow.s['wageK'][40], 0); put('SlowMWageChangeTenSigned', slow.s['wageM'][40], 0)
    put('AncOutputBand', np.abs(a['output']).max())
    # where the residents outside the private workforce are, and the same for the initial knowledge cohort
    for tag, q in (('Six', q6), ('Long', qL)):
        for name, key in (('Inactive', 'inactive'), ('Retired', 'retired'), ('OtherWork', 'otherwork')):
            put('Outside' + name + 'Rise' + tag, m[key][q] - main.start[key], 1)
        put('OutsideOtherWorkShare' + tag, 100 * (m['otherwork'][q] - main.start['otherwork']) / (m['outside'][q] - main.start['outside']))
        put('CohUnempGap' + tag, main.cohort[q, 2] - main.cohort_ref[q, 2], 1)
        put('CohWorkShare' + tag, 100 * (main.cohort[q, 1] - main.cohort_ref[q, 1] + main.cohort_out[q, 2]) / (main.cohort_ref[q, 0] - main.cohort[q, 0]))
        for j, name in enumerate(('Inactive', 'Retired', 'OtherWork')):
            put('Coh' + name + 'Gap' + tag, main.cohort_out[q, j], 1)
    put('RobOtherWorkShareMin', min(100 * (c.s['otherwork'][qL] - c.start['otherwork']) / (c.s['outside'][qL] - c.start['outside']) for c in every))
    put('RobOtherWorkShare', np.mean([100 * (c.s['otherwork'][qL] - c.start['otherwork']) / (c.s['outside'][qL] - c.start['outside']) for c in every]))
    put('RobOtherWorkShareMax', max(100 * (c.s['otherwork'][qL] - c.start['otherwork']) / (c.s['outside'][qL] - c.start['outside']) for c in every))
    # the experiment, the second state of demand and the variant of the return, at the same dates
    for prefix, case in (('', main), ('Pol', pol), ('Anc', anc), ('Tra', tra), ('Pur', pur)):
        for tag, q in (('Six', q6), ('Ten', 40), ('Long', qL)):
            put(prefix + 'CohNoWorkGap' + tag, nowork(case, q), 1)
            if prefix:
                put(prefix + 'CohOtherWorkGap' + tag, case.cohort_out[q, 2], 1)
            put(prefix + 'RegimeUse' + tag, case.s['use'][q])
            put(prefix + 'RegimeMJobsPct' + tag, 100 * (case.s['jobsM'][q] / case.start['jobsM'] - 1), 1)
    put('CapacityRiseLong', m['capacity'][qL], 1); put('CapacityRiseSix', m['capacity'][q6], 1)
    put('CapConsFallMax', -m['capcons'].min()); put('AICapitalLong', m['aicap'][qL], 1); put('AICapitalSix', m['aicap'][q6], 1)
    put('KJobsStart', main.start['jobsK'], 1); put('MJobsStart', main.start['jobsM'], 1)
    put('KJobsLostLong', main.start['jobsK'] - m['jobsK'][qL], 1); put('MJobsGainedLong', m['jobsM'][qL] - main.start['jobsM'], 1)
    put('UnempStart', main.start['unemp'], 1); put('UnempPeak', m['unemp'].max(), 1); put('OutsideStart', main.start['outside'], 1)
    put('FindKStart', main.start['findK'], 2); put('FindKMin', m['findK'].min(), 2); put('FindKLong', m['findK'][qL], 2)
    put('FindMStart', main.start['findM'], 2); put('FindMMin', m['findM'].min(), 2); put('FindMLong', m['findM'][qL], 2)
    put('TrainStart', main.start['train']); put('TrainPeak', m['train'].max(), -1); put('QualStart', main.start['qualified'], 1)
    put('ReturnRef', 100 * (main.eco.R ** 4 - 1), 1); put('ReturnMax', m['ret'].max(), 1)
    put('LayoffsMax', main.p['layoffs'].max() * main.sc, 1)
    # the initial knowledge cohort by qualification and by savings at the initial date (main scenario)
    for name, idx in Case.GROUPS.items():
        put('Grp' + name + 'Share', main.group(idx, 0)['share'])
        for tag, q in (('Six', q6), ('Ten', 40), ('Long', qL)):
            g = main.group(idx, q)
            put('Grp' + name + 'KGap' + tag, g['outK'], 1); put('Grp' + name + 'MGap' + tag, g['intoM'], 1)
            put('Grp' + name + 'OtherWorkGap' + tag, g['other'], 1); put('Grp' + name + 'NoWorkGap' + tag, g['nowork'], 1)
            put('Grp' + name + 'ConsFall' + tag, -g['cons'])
    put('CohConsFallSix', -main.group(list(range(9)), q6)['cons'])     # consumption per surviving member of the whole cohort, as in Table 7
    for tag, q in (('Six', q6), ('Ten', 40), ('Long', qL)):
        g = main.group(Case.GROUPS['NoQual'], q)
        put('GrpNoQualQualified' + tag, g['qualified']); put('GrpNoQualQualifiedRef' + tag, g['qualifiedref'])
        put('GrpNoQualQualifiedRise' + tag, g['qualified'] - g['qualifiedref'])
        put('GrpNoQualMJob' + tag, g['jobM'], 1); put('GrpNoQualMJobRef' + tag, g['jobMref'], 1)
        g = main.group(Case.GROUPS['Qual'] + Case.GROUPS['ManualHome'], q)
        put('GrpCanKGap' + tag, g['outK'], 1); put('GrpCanMGap' + tag, g['intoM'], 1); put('GrpCanNoWorkGap' + tag, g['nowork'], 1)
    for tag, q in (('Six', q6), ('Ten', 40), ('Long', qL)):               # points of the whole cohort moved into manual work by the unqualified
        g = main.group(Case.GROUPS['NoQual'], q)
        put('GrpNoQualMContrib' + tag, g['share'] * g['intoM'] / 100, 1)
    put('GrpManualHomeLowShare', 100 * main.group([6], 0)['share'] / main.group(Case.GROUPS['ManualHome'], 0)['share'])
    put('GrpHighSaveNoQualShare', 100 * main.group([2], 0)['share'] / main.group(Case.GROUPS['HighSave'], 0)['share'])
    put('GrpCanShare', main.group(Case.GROUPS['Qual'] + Case.GROUPS['ManualHome'], 0)['share'])
    # largest difference between the thirds of savings inside a class of qualification (knowledge-home members)
    inside = lambda key: max(max(main.group([3 * c + j], q)[key] for j in range(3)) - min(main.group([3 * c + j], q)[key] for j in range(3))
                             for c in (0, 1) for q in (q6, 40, qL))
    put('GrpSaveSpread', max(inside(k) for k in ('outK', 'intoM', 'other', 'nowork')), 1); put('GrpSaveSpreadCons', inside('cons'), 1)
    put('GrpSaveSpreadNoWork', inside('nowork'), 1)
    put('GrpQualLowMGapSix', main.group([3], q6)['intoM'], 1); put('GrpQualHighMGapSix', main.group([5], q6)['intoM'], 1)
    P0, mass = main.eco.P, tr.start_state(main.eco, main.ref)['groups']
    earn = (sum(mass[:, h, :, emp(N)].sum() * P0.eff[h, N] for h in range(P0.nh)) * (main.eco.wage_share * main.ref['lam'])[N]
            / mass[:, :, :, emp(N)].sum())                  # quarterly earnings of an employed member of the cohort
    for name in ('LowSave', 'MidSave', 'HighSave'):
        x = mass[Case.GROUPS[name]].sum((0, 1, 2, 3))
        put('Grp' + name + 'Quarters', x @ P0.agrid / x.sum() / earn, 1)
    # easier access to manual work under two states of demand (Section 7.4); all against the common reference
    def new(name, value, d=0):
        assert name not in out, name
        put(name, value, d)
    for prefix, case in (('', main), ('Anc', anc), ('Acc', acc), ('AncAcc', ancacc)):
        share, share_ref = case.home_manual()
        new(prefix + 'FindMAccSix', case.s['findM'][q6], 2); new(prefix + 'HomeMJobSix', share[q6], 1)
        new(prefix + 'MJobsMillionSix', case.s['jobsM'][q6], 1); new(prefix + 'CohNoJobShareSix', case.cohort[q6, 4], 1)
        if prefix in ('Acc', 'AncAcc'):
            for tag, q in (('Six', q6), ('Long', qL)):
                new(prefix + 'CohKGap' + tag, case.cohort_ref[q, 0] - case.cohort[q, 0], 1)
                new(prefix + 'CohMGap' + tag, case.cohort[q, 1] - case.cohort_ref[q, 1], 1)
                new(prefix + 'CohNoWorkGap' + tag, nowork(case, q), 1)
                new(prefix + 'MWageChange' + tag, case.s['wageM'][q], 0); new(prefix + 'MWageFall' + tag, -case.s['wageM'][q])
    new('HomeMJobRef', main.home_manual()[1], 1)
    shift = lambda a, b, j: (a.cohort[q6, j] - a.cohort_ref[q6, j]) - (b.cohort[q6, j] - b.cohort_ref[q6, j])
    pairs = (('Acc', acc, main), ('AncAcc', ancacc, anc))
    for prefix, a, b in pairs:
        new(prefix + 'MShiftSix', shift(a, b, 1), 1); new(prefix + 'NoWorkShiftSix', abs(nowork(b, q6) - nowork(a, q6)), 1)
        new(prefix + 'MWageGapSix', b.s['wageM'][q6] - a.s['wageM'][q6]); new(prefix + 'HomeMJobFallSix', b.home_manual()[0][q6] - a.home_manual()[0][q6], 1)
    new('AccMShiftMinSix', min(shift(a, b, 1) for _, a, b in pairs), 1); new('AccMShiftMaxSix', max(shift(a, b, 1) for _, a, b in pairs), 1)
    new('AccNoWorkShiftMaxSix', max(abs(nowork(b, q6) - nowork(a, q6)) for _, a, b in pairs), 1)
    new('AccNoWorkLowerCases', sum(nowork(a, q6) < nowork(b, q6) for _, a, b in pairs))
    # easier access without a rise in capability (the control), and capability measured against it (the experiment only)
    kgap = lambda c: c.cohort_ref[:, 0] - c.cohort[:, 0]
    mgap = lambda c: c.cohort[:, 1] - c.cohort_ref[:, 1]
    wgap = lambda c: c.cohort[:, 2] - c.cohort_ref[:, 2] + c.cohort_out[:, 0] + c.cohort_out[:, 1]
    for name, fn in (('CohK', kgap), ('CohM', mgap), ('NoWork', wgap), ('OtherWork', lambda c: c.cohort_out[:, 2])):
        new('CtlAvg' + name, avg(fn(ctl)), 1); new('MatAvg' + name, avg(fn(acc) - fn(ctl)), 1)
        new('CtlLong' + name, fn(ctl)[qL], 1); new('MatLong' + name, (fn(acc) - fn(ctl))[qL], 1)
    against = lambda key: 100 * (acc.p[key] / ctl.p[key] - 1)        # paths of the experiment with easier access against the control
    new('MatAvgMWageFall', -avg(against('wage')[:, M])); new('MatAvgKWageFall', -avg(against('wage')[:, N]))
    new('MatLongMWageFall', -against('wage')[qL, M]); new('MatAvgOutputFall', -avg(against('output'))); new('MatAvgConsFall', -avg(against('household_consumption')))
    new('MatAvgMJobsPct', avg(against('employment')[:, M])); new('MatAvgKJobsFallPct', -avg(against('employment')[:, N]))
    actl = ancacc.control                                    # the control when the anchor follows income (Section 7.3)
    for name, fn in (('CohK', kgap), ('CohM', mgap), ('NoWork', wgap), ('OtherWork', lambda c: c.cohort_out[:, 2])):
        new('AncCtlAvg' + name, avg(fn(actl)), 1); new('AncMatAvg' + name, avg(fn(ancacc) - fn(actl)), 1)
        new('AncCtlLong' + name, fn(actl)[qL], 1); new('AncMatLong' + name, (fn(ancacc) - fn(actl))[qL], 1)
    aagainst = lambda key: 100 * (ancacc.p[key] / actl.p[key] - 1)
    new('AncMatAvgMWageFall', -avg(aagainst('wage')[:, M])); new('AncMatLongMWageFall', -aagainst('wage')[qL, M]); new('AncMatAvgMJobsPct', avg(aagainst('employment')[:, M]))
    new('AncCtlAvgMWageFall', -avg(actl.s['wageM'])); new('AncCtlLongMWageFall', -actl.s['wageM'][qL]); new('AncCtlAvgKWageRise', avg(actl.s['wageK']))
    new('AncCtlAvgMJobs', avg(actl.s['jobsM']), 1); new('AncCtlAvgFindM', avg(actl.s['findM']), 2); new('AncCtlAvgHomeM', avg(actl.home_manual()[0]), 1)
    new('AncCtlUseMax', actl.s['use'].max(), 1); new('AncCtlUseMin', actl.s['use'].min(), 1); new('AncCtlAvgOutput', avg(actl.s['output']), 1); new('AncCtlLongOutput', actl.s['output'][qL], 1)
    new('AncCtlLayoffs', actl.p['layoffs'].sum() * actl.sc, 1)
    cs = ctl.s
    new('CtlAvgMJobs', avg(cs['jobsM']), 1); new('CtlAvgMJobsFallPct', -100 * (avg(cs['jobsM']) / ctl.start['jobsM'] - 1))
    new('CtlAvgKJobs', avg(cs['jobsK']), 1)
    new('CtlAvgLMRise', 100 * (avg(ctl.p['labour'][:, M]) / ctl.ag['labour'][M] - 1)); new('CtlAvgMWageFall', -avg(cs['wageM'])); new('CtlLongMWageFall', -cs['wageM'][qL])
    new('CtlAvgKWageRise', avg(cs['wageK'])); new('CtlAvgFindM', avg(cs['findM']), 2); new('CtlAvgHomeM', avg(ctl.home_manual()[0]), 1)
    new('CtlAvgUseOver', avg(cs['use']) - 100, 1); new('CtlUseMax', cs['use'].max()); new('CtlAvgOutputRise', avg(cs['output']), 1); new('CtlLayoffs', ctl.p['layoffs'].sum() * ctl.sc, 1)
    # the policy experiments: a transfer of licence rent and public purchases (Section 9)
    for prefix, case in (('Tra', tra), ('Pur', pur)):
        v = case.s
        for tag, q in (('Six', q6), ('Ten', 40), ('Long', qL)):
            new(prefix + 'OutputChange' + tag, abs(v['output'][q])); new(prefix + 'ConsChange' + tag, abs(v['cons'][q]))
            new(prefix + 'KWageChange' + tag, abs(v['wageK'][q])); new(prefix + 'MWageChange' + tag, abs(v['wageM'][q]))
            new(prefix + 'UseAt' + tag, v['use'][q]); new(prefix + 'KJobsAt' + tag, v['jobsK'][q], 1); new(prefix + 'MJobsAt' + tag, v['jobsM'][q], 1)
            new(prefix + 'OutsideChange' + tag, abs(v['outside'][q] - case.start['outside']), 1)
            new(prefix + 'CohKGap' + tag, case.cohort_ref[q, 0] - case.cohort[q, 0], 1); new(prefix + 'CohMGap' + tag, case.cohort[q, 1] - case.cohort_ref[q, 1], 1)
            new(prefix + 'CapConsChange' + tag, abs(v['capcons'][q])); new(prefix + 'WageBillChange' + tag, abs(v['wagebill'][q]))
            new(prefix + 'MachinesChange' + tag, abs(v['machines'][q])); new(prefix + 'CapacityChange' + tag, abs(v['capacity'][q]))
            new(prefix + 'Transfer' + tag, v['transfer'][q], 1); new(prefix + 'Purchases' + tag, v['purchases'][q])
            new(prefix + 'PurchasesShare' + tag, 100 * case.p['purchases'][q] / case.p['output'][q])
            new(prefix + 'TransferOutput' + tag, 100 * case.p['transfer'][q] / case.acc['output'], 1)
        new(prefix + 'UnempMax', v['unemp'].max(), 1); new(prefix + 'UnempMin', v['unemp'].min(), 1)
        new(prefix + 'UseMin', v['use'].min()); new(prefix + 'UseMax', v['use'].max())
        new(prefix + 'AvgTransfer', avg(v['transfer']), 1); new(prefix + 'AvgPurchases', avg(v['purchases']))
        new(prefix + 'AvgKJobsFallPct', -100 * (avg(v['jobsK']) / case.start['jobsK'] - 1)); new(prefix + 'AvgWageBillFall', -avg(v['wagebill']))
        new(prefix + 'Layoffs', case.p['layoffs'].sum() * case.sc, 1)
        gap = np.array([nowork(case, q) for q in range(qL + 1)])
        new(prefix + 'NoWorkPeak', gap.max(), 1); new(prefix + 'NoWorkPeakYear', int(gap.argmax()) // 4 + 1)
    for prefix, case in (('Tra', tra), ('Pur', pur)):           # years in which output is above the reference, and its peak
        v = case.s
        above = np.where(v['output'][:qL + 1] > 0)[0]
        new(prefix + 'AboveFirst', (above.min() // 4 + 1) if len(above) else 0); new(prefix + 'AboveLast', (above.max() // 4 + 1) if len(above) else 0)
        new(prefix + 'OutputPeak', v['output'].max(), 1); new(prefix + 'OutputPeakYear', int(v['output'].argmax()) // 4 + 1)
        new(prefix + 'UsePeakYear', int(v['use'].argmax()) // 4 + 1)
    # the limit case: capacity after forty years in three steps (the tasks that AI performs, conventional capital, labour)
    def capacity_steps(case):
        t, al, ag = case.eco.tech, case.ref['allocation'], case.ag
        def level(front, KC, KA, LN, LM):
            x = prod.quantities(t, np.array([front]), np.array([KC]), np.array([KA]), np.array([LN]), np.array([LM]), np.array([t.dep_A * KA]))
            return 100 * (float(x['Y'][0]) / case.acc['output'] - 1)
        KC, KA = case.p['capital'][qL]; LN, LM = case.p['labour'][qL]; front = float(case.p['frontier'][qL])
        a = level(front, al.KC, KA, ag['labour'][N], ag['labour'][M]); b = level(front, KC, KA, ag['labour'][N], ag['labour'][M])
        c = level(front, KC, KA, LN, LM)
        return dict(tech=a, capital=b - a, labour=c - b, total=c, LM=100 * (LM / ag['labour'][M] - 1), LN=100 * (LN / ag['labour'][N] - 1),
                    KY=100 * ((KC / case.p['output'][qL]) / (al.KC / case.acc['output']) - 1))
    dec, dec0 = capacity_steps(pur), capacity_steps(main)
    new('DecTech', dec['tech']); new('DecCapital', dec['capital']); new('DecLabour', dec['labour']); new('DecTotal', dec['total'])
    new('DecLMRise', dec['LM']); new('DecLNFall', -dec['LN']); new('DecKYRise', dec['KY']); new('DecTechMain', dec0['tech'])
    new('PurAICapitalLong', pur.s['aicap'][qL], 1); new('PurOverQuarters', int((pur.s['use'] > 100.0001).sum())); new('PurOverYears', int((pur.s['use'] > 100.0001).sum()) // 4)
    # public purchases: capability raises output, pay and employment, so these are stated as rises
    pv = pur.s
    new('PurUseOver', pv['use'].max() - 100); new('PurAvgMWageRise', avg(pv['wageM'])); new('PurAvgKWageRise', avg(pv['wageK']))
    new('PurAvgConsRise', avg(pv['cons'])); new('PurAvgOutputRise', avg(pv['output'])); new('PurAvgWageBillRise', avg(pv['wagebill']))
    new('PurAvgFewer', -avg([nowork(pur, q) for q in range(FIRST)]), 1); new('PurLongFewer', -nowork(pur, qL), 1)
    new('PurCapConsPeak', pv['capcons'].max()); new('PurMJobsRisePctLong', 100 * (pv['jobsM'][qL] / pur.start['jobsM'] - 1))
    new('PurKJobsFallPctLong', -100 * (pv['jobsK'][qL] / pur.start['jobsK'] - 1))
    new('PurPvCohRise', consumption_loss(pur), 1); new('PurPvHouseRise', present_value(pur, pur.p['household_consumption'], pur.acc['household_consumption']), 1)
    new('PurPvWageBillRise', present_value(pur, pur.p['payroll'], pur.ag['payroll'])); new('PurPvOutputRise', present_value(pur, pur.p['output'], pur.acc['output']))
    new('PurSolverStepMin', min(x.get('solver_step', 0.05) for x in json.load(open(OUT / 'transition_purchases.json'))['rows'][0]['steps']), 2)
    unit = lambda key: (tra.p[key][qL] - main.p[key][qL]) / tra.p['transfer'][qL]       # long-run effect of a unit of transfer
    new('TraConsPerUnit', unit('household_consumption'), 1); new('TraOutputPerUnit', unit('output'), 1); new('TraWageBillPerUnit', unit('payroll'), 1)
    new('TraConsChangeLongOne', -tra.s['cons'][qL], 1); new('TraConsGainLong', tra.s['cons'][qL] - main.s['cons'][qL], 1)
    first = np.argmax(pur.p['purchases'] > 1e-9) if (pur.p['purchases'] > 1e-9).any() else 0
    new('PurFirstYear', int(first) // 4 + 1); new('PurReturnSpread', pur.s['ret'].max() - pur.s['ret'].min(), 3)
    # the boom when the return follows the use of capacity, and three tests of it (Section 9.1 and Appendix D)
    booms = {prefix: boom(case) for prefix, case in zip(('Mon', 'MonFore', 'MonLow', 'MonHigh'), (pol,) + tuple(tests))}
    for prefix, b in booms.items():
        new(prefix + 'PeakOutput', b['peak']); new(prefix + 'PeakYear', b['year']); new(prefix + 'PeakInvest', b['invest']); new(prefix + 'PeakCapCons', b['capcons'])
        new(prefix + 'PeakHouseholds', b['households'], 1); new(prefix + 'Slack', b['slack']); new(prefix + 'Zero', b['zero'])
        new(prefix + 'NoWork', b['nowork'], 1); new(prefix + 'Worst', b['worst'], 1); new(prefix + 'WorstYear', b['worstyear'])
    new('MonInvestMin', min(b['invest'] for b in booms.values())); new('MonInvestMax', max(b['invest'] for b in booms.values()))
    new('MonNoWorkMin', min(b['nowork'] for b in booms.values()), 1); new('MonNoWorkMax', max(b['nowork'] for b in booms.values()), 1)
    new('MonZeroMin', min(b['zero'] for b in booms.values())); new('MonZeroMax', max(b['zero'] for b in booms.values()))
    new('MonLongGap', max(abs(c.s[k][qL] - pol.s[k][qL]) for c in tests for k in ('output', 'cons', 'wageK', 'wageM')), 1)
    new('MonHouseholdsMax', max(abs(b['households']) for b in booms.values()), 1)
    new('RefCapConsOutputShare', 100 * main.acc['capitalist_consumption'] / main.acc['output'])
    new('PolCapIncomeRiseSix', pol.s['capincome'][q6]); new('PurCapIncomePeak', pur.s['capincome'].max())
    # further checks (Appendix E)
    for prefix, case in (('Eta', checks[0]), ('Cost', checks[1]), ('Whole', checks[2])):
        v = case.s
        for tag, q in (('Six', q6), ('Long', qL)):
            new(prefix + 'OutputFall' + tag, -v['output'][q]); new(prefix + 'ConsFall' + tag, -v['cons'][q])
            new(prefix + 'KWageFall' + tag, -v['wageK'][q]); new(prefix + 'MWageFall' + tag, -v['wageM'][q])
            new(prefix + 'UseAt' + tag, v['use'][q]); new(prefix + 'KJobsAt' + tag, v['jobsK'][q], 1); new(prefix + 'MJobsAt' + tag, v['jobsM'][q], 1)
            new(prefix + 'NoWorkAt' + tag, nowork(case, q), 1)
        new(prefix + 'UnempMax', v['unemp'].max(), 1); new(prefix + 'UseMax', v['use'].max()); new(prefix + 'Layoffs', case.p['layoffs'].sum() * case.sc, 1)
        new(prefix + 'OutputMax', v['output'].max(), 1); new(prefix + 'AICapitalLong', v['aicap'][qL], 1)
        new(prefix + 'PvCohFall', -consumption_loss(case), 1)
        new(prefix + 'PvHouseFall', -present_value(case, case.p['household_consumption'], case.acc['household_consumption']), 1)
    # the experiment without applications of employed knowledge workers for manual jobs (Appendix E)
    napp, nacc, nctl = main.noapp
    apps0, apps1 = main.ref['households']['ag']['applications'][M], napp.ref['households']['ag']['applications'][M]
    onjob = sum(main.eco.P.offer_intensity[N] * (1 - main.eco.P.exit_E[h].sum()) * main.ref['households']['mu'][h, 1, hh.emp(N)].sum()
                for h in range(main.eco.P.nh) if hh.accessible(main.eco.P, h, 1, M))
    new('RefAppsOnJobShare', 100 * onjob / apps0); new('NappAppsRatio', apps0 / apps1, 1)
    new('RefVacancyCostRatio', main.eco.vacancy_cost[N] / main.eco.vacancy_cost[M]); new('NappVacancyCostRatio', napp.eco.vacancy_cost[N] / napp.eco.vacancy_cost[M], 1)
    new('NappRefFlowEmpNM', run.fit(napp.const, napp.ref)['N to M, employed origin, % per quarter'], 2)
    for prefix, case in (('Napp', napp), ('NappAcc', nacc), ('NappCtl', nctl)):
        v = case.s
        new(prefix + 'AvgCohK', avg(kgap(case)), 1); new(prefix + 'AvgCohM', avg(mgap(case)), 1); new(prefix + 'AvgNoWork', avg(wgap(case)), 1)
        new(prefix + 'LongNoWork', wgap(case)[qL], 1); new(prefix + 'AvgManualShare', 100 * avg(mgap(case)) / avg(kgap(case)))
        new(prefix + 'AvgMWageFall', -avg(v['wageM'])); new(prefix + 'LongMWageFall', -v['wageM'][qL]); new(prefix + 'AvgFindM', avg(v['findM']), 2)
        new(prefix + 'AvgMJobsPct', 100 * (avg(v['jobsM']) / case.start['jobsM'] - 1), 1); new(prefix + 'AvgHomeM', avg(case.home_manual()[0]), 1)
        new(prefix + 'AvgOutputFall', -avg(v['output'])); new(prefix + 'LongOutputFall', -v['output'][qL]); new(prefix + 'Layoffs', case.p['layoffs'].sum() * case.sc, 1)
    new('NappAccNoWorkGain', avg(wgap(napp)) - avg(wgap(nacc)), 1); new('DemandNoWorkGap', avg(wgap(main)) - avg(wgap(anc)), 1)
    new('NappMatAvgNoWork', avg(wgap(nacc) - wgap(nctl)), 1); new('NappMatAvgCohM', avg(mgap(nacc) - mgap(nctl)), 1)
    new('AccAvgManualShareAll', 100 * avg(mgap(acc)) / avg(kgap(acc)))
    # other earnings of knowledge-home residents in manual work, and a lower share qualified without instruction (Appendix E)
    fitof = lambda c: run.fit(c.const, c.ref)
    for prefix, trio in (('RhoLow', main.rholow), ('RhoHigh', main.rhohigh), ('AccLow', (main.acclow, None, None))):
        c, a, k = trio
        new(prefix + 'RefKShare', fitof(c)['knowledge employment share, %'], 1); new(prefix + 'RefFlowNM', fitof(c)['N to M, unemployed origin, % per quarter'], 1)
        new(prefix + 'RefFlowEmpNM', fitof(c)['N to M, employed origin, % per quarter'], 2)
        new(prefix + 'AvgNoWork', avg(wgap(c)), 1); new(prefix + 'AvgCohM', avg(mgap(c)), 1); new(prefix + 'AvgCohK', avg(kgap(c)), 1)
        new(prefix + 'AvgManualShare', 100 * avg(mgap(c)) / avg(kgap(c))); new(prefix + 'AvgMWageFall', -avg(c.s['wageM'])); new(prefix + 'LongMWageFall', -c.s['wageM'][qL])
        new(prefix + 'LongNoWork', wgap(c)[qL], 1); new(prefix + 'AvgOutputFall', -avg(c.s['output'])); new(prefix + 'Layoffs', c.p['layoffs'].sum() * c.sc, 1)
        if a is not None:
            new(prefix + 'AccAvgNoWork', avg(wgap(a)), 1); new(prefix + 'AccAvgMWageFall', -avg(a.s['wageM'])); new(prefix + 'AccAvgHomeM', avg(a.home_manual()[0]), 1)
            new(prefix + 'CtlAvgMWageFall', -avg(k.s['wageM'])); new(prefix + 'CtlAvgHomeM', avg(k.home_manual()[0]), 1); new(prefix + 'CtlAvgFindM', avg(k.s['findM']), 2)
            new(prefix + 'MatAvgNoWork', avg(wgap(a) - wgap(k)), 1)
    every5 = (main, napp, main.rholow[0], main.rhohigh[0], main.acclow)
    new('AllNoWorkMin', min(avg(wgap(c)) for c in every5), 1); new('AllNoWorkMax', max(avg(wgap(c)) for c in every5), 1)
    new('AllOffsetMin', min(100 * avg(mgap(c)) / avg(kgap(c)) for c in every5)); new('AllOffsetMax', max(100 * avg(mgap(c)) / avg(kgap(c)) for c in every5))
    new('AllMWageFallMin', min(-avg(c.s['wageM']) for c in every5)); new('AllMWageFallMax', max(-avg(c.s['wageM']) for c in every5))
    new('RhoNoWorkMin', min(avg(wgap(c)) for c in (main, main.rholow[0], main.rhohigh[0])), 1); new('RhoNoWorkMax', max(avg(wgap(c)) for c in (main, main.rholow[0], main.rhohigh[0])), 1)
    new('RhoOffsetMin', min(100 * avg(mgap(c)) / avg(kgap(c)) for c in (main, main.rholow[0], main.rhohigh[0]))); new('RhoOffsetMax', max(100 * avg(mgap(c)) / avg(kgap(c)) for c in (main, main.rholow[0], main.rhohigh[0])))
    gains = [avg(wgap(x)) - avg(wgap(y)) for x, y in ((main, acc), (main.rholow[0], main.rholow[1]), (main.rhohigh[0], main.rhohigh[1]))]
    new('RhoAccNoWorkGainMax', max(gains), 1)
    whole = checks[2]
    new('WholeTransferLong', whole.s['transfer'][qL], 1); new('WholeAvgTransfer', avg(whole.s['transfer']), 1)
    unit2 = lambda key: (whole.p[key][qL] - main.p[key][qL]) / whole.p['transfer'][qL]
    new('WholeConsPerUnit', unit2('household_consumption'), 1); new('WholeOutputPerUnit', unit2('output'), 1)
    new('WholePvCapChange', present_value(whole, whole.p['capitalist_consumption'], whole.acc['capitalist_consumption']), 1)
    new('CostRentRiseLong', 100 * (checks[1].p['licence_rent'][qL] / checks[1].p['licence_rent'][0] - 1) if 'licence_rent' in checks[1].p.files else 0.0)
    new('CostAIShareLong', checks[1].s['ai'][qL])
    for prefix, case in (('', main), ('Cost', checks[1])):
        new(prefix + 'LabourPriceFallLong', 100 * (1 - case.p['lam'][qL, 0] / case.ref['lam'][0])); new(prefix + 'AIPriceRiseLong', 100 * (np.ravel(case.p['ai_price'])[qL] - 1))
    def task_share(case):                                    # share of knowledge tasks in the cost of tasks, percent
        p = case.p; m = np.ravel(p['automated'])
        k = p['lam'][:, 0] * p['labour'][:, 0] / (1 - m); x = p['lam'][:, 1] * p['labour'][:, 1]
        return 100 * (k / (k + x))[:qL + 1]
    share = task_share(main)
    new('TaskShareRef', task_share(checks[0]).mean(), 1); new('TaskShareMin', share.min(), 1); new('TaskShareMax', share.max(), 1)
    new('WholeSlackYear', int(np.argmax(whole.s['use'] < 100)) // 4 + 1); new('WholeBelowYear', int(np.argmax(whole.s['output'] < 0)) // 4 + 1)
    new('WholeOutputPeakYear', int(whole.s['output'].argmax()) // 4 + 1); new('WholeUsePeak', whole.s['use'].max(), 1)
    new('WholeNoWorkWorst', max(nowork(whole, q) for q in range(qL + 1)), 1)
    # where the fall of output goes in the labour market (Section 6.3), and the same with slower wages (Section 8.4)
    for prefix, case in (('', main), ('Slow', slow)):
        v = case.s
        for tag, q in (('Six', q6), ('Long', qL)):
            new(prefix + 'JobsFall' + tag, -v['jobs'][q], 1); new(prefix + 'SuppliedFall' + tag, -v['supplied'][q])
            new(prefix + 'UsedFall' + tag, -v['used'][q]); new(prefix + 'AnyWork' + tag, v['anywork'][q], 1)
            new(prefix + 'SuppliedFall' + tag + 'One', -v['supplied'][q], 1); new(prefix + 'UsedFall' + tag + 'One', -v['used'][q], 1)
        new(prefix + 'AnyWorkMin', v['anywork'].min(), 1)
    new('AnyWorkRef', main.start['anywork'], 1); new('SlowUseSix', slow.s['use'][q6]); new('SlowWageBillFallSix', -slow.s['wagebill'][q6])
    new('SlowUseLong', slow.s['use'][qL])
    Pm = main.eco.P
    new('ExitUMin', 100 * Pm.exit_U.sum(1).min()); new('ExitUMax', 100 * Pm.exit_U.sum(1).max())
    new('ExitEMin', 100 * Pm.exit_E.sum(1).min(), 1); new('ExitEMax', 100 * Pm.exit_E.sum(1).max(), 1)
    new('SlowPvCohFall', -consumption_loss(slow), 1)
    new('SlowPvHouseFall', -present_value(slow, slow.p['household_consumption'], slow.acc['household_consumption']), 1)
    new('SlowPvWageBillFall', -present_value(slow, slow.p['payroll'], slow.ag['payroll']))
    # the present value of consumption over forty years in the cases that bear on policy (Section 9)
    for prefix, case in (('', main), ('Anc', anc), ('Acc', acc), ('Own', own), ('Pol', pol), ('Tra', tra), ('Pur', pur)):
        new(prefix + 'PvCohFall', -consumption_loss(case), 1)
        new(prefix + 'PvHouseFall', -present_value(case, case.p['household_consumption'], case.acc['household_consumption']), 1)
        new(prefix + 'PvCapChange', abs(present_value(case, case.p['capitalist_consumption'], case.acc['capitalist_consumption'])), 1 if prefix in ('', 'Acc', 'Own', 'Tra', 'Pur') else 0)
        new(prefix + 'PvWageBillFall', -present_value(case, case.p['payroll'], case.ag['payroll']))
    thirds = [-consumption_loss(main, Case.GROUPS[k]) for k in ('LowSave', 'MidSave', 'HighSave')]
    new('PvSaveFallMin', min(thirds), 1); new('PvSaveFallMax', max(thirds), 1)
    for name in ('NoQual', 'Qual', 'ManualHome'):
        new('PvGrp' + name + 'Fall', -consumption_loss(main, Case.GROUPS[name]), 1)
    # welfare: consumption-equivalent changes of residents alive at the initial date (Section 9 and Section 6)
    for prefix, case in (('', main), ('Anc', anc), ('Acc', acc), ('Own', own), ('Pol', pol), ('Tra', tra), ('Pur', pur), ('Slow', slow),
                         ('Eta', checks[0]), ('Cost', checks[1]), ('Whole', checks[2])):
        if case.W is None:
            continue
        groups = case.residents()
        for name in ('All', 'Cohort', 'Manual', 'Inactive', 'Retired', 'OtherWork', 'NoQual', 'Qual', 'ManualHome', 'ManualK', 'ManualM'):
            new(prefix + 'Wf' + name, abs(case.welfare(groups[name])), 1)
        thirds = [case.welfare(groups[k]) for k in ('LowSave', 'MidSave', 'HighSave')]
        new(prefix + 'WfSaveMin', min(abs(x) for x in thirds), 1); new(prefix + 'WfSaveMax', max(abs(x) for x in thirds), 1)
        if case.more is not None:                       # the same at the return on claims, and residents who enter later
            for name in ('All', 'Cohort', 'Manual', 'Inactive', 'Retired', 'OtherWork', 'NoQual', 'Qual', 'ManualHome', 'ManualK', 'ManualM'):
                new(prefix + 'Ws' + name, abs(case.social(groups[name])), 1)
            for tag, q in (('Ten', 40), ('Twenty', 80), ('Long', qL)):
                new(prefix + 'En' + tag, abs(case.entrant(q)), 1); new(prefix + 'Es' + tag, abs(case.entrant(q, True)), 1)
    # easier access and welfare: access alone (the control), capability at unchanged access, capability at easier access
    # (from the path with easier access and constant capability, as the ratio of the two consumption equivalents)
    if ctl.W is not None and ctl.more is not None:
        both = lambda social: (lambda case, g: (case.social if social else case.welfare)(case.residents()[g]))
        for kind, social in (('Wf', False), ('Ws', True)):
            fn = both(social)
            for name in ('All', 'Cohort', 'NoQual', 'Qual', 'Manual', 'ManualK', 'ManualM', 'Inactive'):
                alone, joint = fn(ctl, name), fn(acc, name)
                new('Ctl' + kind + name, abs(alone), 1); new('Mat' + kind + name, abs(100 * ((1 + joint / 100) / (1 + alone / 100) - 1)), 1)
    P_ = main.eco.P; ent = main.more['entrants']
    new('BetaLow', P_.beta.min(), 3); new('BetaHigh', P_.beta.max(), 3); new('BetaLowYear', P_.beta.min() ** 4, 2); new('BetaHighYear', P_.beta.max() ** 4, 2)
    new('BetaMarket', main.more['beta'], 2); new('BetaMarketYear', main.more['beta'] ** 4, 2)
    new('EntrantInactive', 100 * ent[:, :, PRIVATE].sum() / ent.sum()); new('EntrantNoClaims', 100 * ent[..., 0].sum() / ent.sum())
    new('EntrantConsShare', 100 * ((ent * np.where(ent > 0, main.ref['households']['sol']['policy']['cons'], 0.0)).sum() / ent.sum())
        / (main.acc['household_consumption'] / main.ref['households']['mu'].sum()))
    new('TraEntrantShare', 100 * (tra.p['transfer'][qL] / main.ref['households']['mu'].sum())
        / ((ent * np.where(ent > 0, main.ref['households']['sol']['policy']['cons'], 0.0)).sum() / ent.sum()))
    new('PolUseOver', pol.s['use'].max() - 100); new('PolOverYears', int((pol.s['use'] > 100.0001).sum()) // 4)
    new('TraUseOver', tra.s['use'].max() - 100, 1); new('TraOverQuarters', int((tra.s['use'] > 100.0001).sum()))   # the transfer: capacity above normal in the first quarters
    if main.W is not None and 'welfare_expected' in main.p.files:      # what the cohort expects at the initial date, before the later steps
        keep, main.W = main.W, np.array(main.p['welfare_expected'])
        new('WfCohortExpected', abs(main.welfare(main.residents()['Cohort'])), 1); main.W = keep
    fit = run.fit(main.const, main.ref)
    put('RefKShare', fit['knowledge employment share, %'], 1); put('DataKShare', par.TARGETS['knowledge employment share, %'], 1)
    put('RefFlowNM', fit['N to M, unemployed origin, % per quarter'], 1); put('DataFlowNM', par.TARGETS['N to M, unemployed origin, % per quarter'], 1)
    put('RefFlowEmpNM', fit['N to M, employed origin, % per quarter'], 2); put('DataFlowEmpNM', par.TARGETS['N to M, employed origin, % per quarter'], 2)
    # offers from the other occupation to employed workers who are qualified for both (Section 5.3): intensity and acceptance in the reference
    P0, mu0, choice = main.eco.P, main.ref['households']['mu'], main.ref['households']['sol']['choice']
    for name, g in (('K', N), ('M', M)):
        mass = np.array([mu0[h, 1, hh.emp(g)].sum() for h in range(P0.nh)])
        taken = np.array([(mu0[h, 1, hh.emp(g)] * choice[(h, 1)][('accept', g)]).sum() if ('accept', g) in choice[(h, 1)] else 0.0 for h in range(P0.nh)])
        put('RefOfferIntensity' + name, P0.offer_intensity[g], 2); put('RefOfferAccept' + name, 100 * taken.sum() / mass.sum(), 2 if g == N else 1)
    jobs = np.array([[mu0[h, :, hh.emp(g)].sum() for g in (N, M)] for h in range(P0.nh)])     # jobs by type and occupation
    put('RefManualJobsKHome', 100 * jobs[P0.home == N, 1].sum() / jobs[:, 1].sum()); put('RefKnowledgeJobsMHome', 100 * jobs[P0.home == M, 0].sum() / jobs[:, 0].sum())
    pay = P0.eff[:, M] * (main.eco.wage_share * main.ref['lam'])[M] * main.const['dollars_per_good'] / 3     # dollars per month in a manual job, by type
    payK = float(jobs[P0.home == N, 1] @ pay[P0.home == N] / jobs[P0.home == N, 1].sum()); payM = float(jobs[P0.home == M, 1] @ pay[P0.home == M] / jobs[P0.home == M, 1].sum())
    put('RefManualPayKHome', payK, -2); put('RefManualPayMHome', payM, -2); put('RefManualPayRatio', payK / payM, 1)
    put('RefManualJobsKHomeRest', 100 - 100 * jobs[P0.home == N, 1].sum() / jobs[:, 1].sum(), 1)
    put('RefManualPayMean', float(jobs[:, 1] @ pay / jobs[:, 1].sum()), -2); put('RefManualJobsKHomeOne', 100 * jobs[P0.home == N, 1].sum() / jobs[:, 1].sum(), 1)
    mass_manual = main.residents(); put('RefManualKShareOfManual', 100 * mass_manual['ManualK'].sum() / mass_manual['Manual'].sum())
    put('Markup', 1 / main.eco.tech.marginal_cost, 2)
    put('TuitionDollars', main.eco.P.tuition * main.const['dollars_per_good'], -2)
    # the two courses that can be chosen, and the value-added target behind the markup, in dollars
    put('TuitionMinDollars', main.eco.P.tuition * main.const['dollars_per_good'] * main.eco.P.course_min, -2)
    put('TuitionMaxDollars', main.eco.P.tuition * main.const['dollars_per_good'] * main.eco.P.course_cap, -2)
    gdp_dollars = main.acc['gdp'] * main.const['dollars_per_good'] * main.const['persons_per_unit']
    put('RefGDPTrillion', gdp_dollars / 1e12, 1)
    put('RefGDPPerResident', gdp_dollars / (main.eco.P.mass.sum() * main.const['persons_per_unit']), -2)
    put('TuitionEarningsRatio', main.eco.P.tuition * main.ag['employment'].sum() / main.ag['payroll'], 1)
    put('HouseholdPayoutShare', 100 * main.ag['claims'], 1)
    put('RefCapConsShare', 100 * main.acc['capitalist_consumption'] / main.acc['gdp'], 1)
    put('RefConsShare', 100 * main.acc['household_consumption'] / main.acc['gdp'], 1)
    put('RefWageShare', 100 * main.ag['payroll'] / main.acc['gdp'], 1)
    put('Population', main.eco.P.mass.sum() * main.sc, 0)
    text = ''.join(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n" for k, v in out.items())
    (GEN / 'numbers.tex').write_text(text, encoding='utf-8')
    return out


def table(name, caption, label, header, rows, notes, spec=None, sep=None, rules='\\midrule'):
    """A full-width table; the first column wraps. `sep` narrows the space between columns (points);
    `rules` is what follows the header line (a full rule unless partial rules are given)."""
    spec = (spec or ('l' + 'r' * (len(header) - 1))).replace('l', '>{\\raggedright\\arraybackslash}X', 1)
    body = ' \\\\\n'.join(' & '.join(r) for r in rows)
    size = '\\small' if sep is None else f'\\small\\setlength{{\\tabcolsep}}{{{sep}pt}}'
    text = (f"\\begin{{table}}[htbp]\\centering{size}\n\\caption{{{caption}}}\\label{{{label}}}\n"
            f"\\begin{{tabularx}}{{\\linewidth}}{{@{{}}{spec}@{{}}}}\\toprule\n{' & '.join(header)} \\\\{rules}\n{body} \\\\\n\\bottomrule\\end{{tabularx}}\n"
            f"\\par\\smallskip\\begin{{minipage}}{{\\linewidth}}\\footnotesize\\emph{{Notes.}} {notes}\\end{{minipage}}\n\\end{{table}}\n")
    (GEN / f'{name}.tex').write_text(plain(text), encoding='utf-8')


def tables(main, pol, mpc, rent, anc, own, slow, acc, ancacc, ctl, tra, pur, tests, checks):
    m = main.s
    head = [''] + [f"{y}" for y, _ in DATES]
    qs = [q for _, q in DATES]
    row = lambda label, key, d=1, sign=True, src=m: [label] + [f(src[key][q], d, sign) for q in qs]
    table('aggregates', 'Capability, spending and output in the experiment', 'tab:aggregates',
          ['Years since the first rise in capability'] + head[1:],
          [row('Knowledge tasks performed by AI (percent)', 'ai', 1, False),
           row('Capacity (percent change)', 'capacity'), row('Output (percent change)', 'output'),
           row('Output relative to capacity (percent)', 'use', 1, False),
           row('Household consumption (percent change)', 'cons'), row('Capitalist consumption (percent change)', 'capcons'),
           row('Wage bill (percent change)', 'wagebill'),
           row('Knowledge wage (percent change)', 'wageK'), row('Manual wage (percent change)', 'wageM'),
           row('Conventional capital (percent change)', 'machines')],
          'Changes are relative to the reference. Wages are posted wages per efficiency unit. The return on claims is '
          f"{f(100 * (main.eco.R ** 4 - 1), 1)} percent per year at every date.")
    table('labor', 'The labor market in the experiment', 'tab:labor',
          ['Years since the first rise in capability'] + head[1:],
          [row('Knowledge jobs (million)', 'jobsK', 1, False), row('Manual jobs (million)', 'jobsM', 1, False),
           row('Unemployment rate (percent)', 'unemp', 1, False),
           row('Residents outside the private workforce (million)', 'outside', 1, False),
           row('\\quad inactive', 'inactive', 1, False), row('\\quad retired', 'retired', 1, False),
           row('\\quad in other employment', 'otherwork', 1, False),
           row('Job finding, knowledge work (per quarter)', 'findK', 2, False),
           row('Job finding, manual work (per quarter)', 'findM', 2, False),
           row('Residents in instruction (thousand per quarter)', 'train', 0, False),
           row('Knowledge-home residents qualified for manual work (percent)', 'qualified', 1, False)],
          f"In the reference there are {f(main.start['jobsK'], 1)} million knowledge jobs and {f(main.start['jobsM'], 1)} million manual "
          f"jobs, the unemployment rate is {f(main.start['unemp'], 1)} percent, {f(main.start['outside'], 1)} million residents are "
          f"outside the private workforce ({f(main.start['inactive'], 1)} inactive, {f(main.start['retired'], 1)} retired and "
          f"{f(main.start['otherwork'], 1)} in other employment, which is public employment and self-employment), job finding is {f(main.start['findK'], 2)} and {f(main.start['findM'], 2)}, "
          f"{f(main.start['train'], 0)} thousand residents are in instruction and {f(main.start['qualified'], 1)} percent of "
          'knowledge-home residents are qualified for manual work.')
    table('adjustment', 'Employment, the use of labor services and pay in the main experiment', 'tab:adjustment',
          ['Years since the first rise in capability'] + head[1:],
          [row('Output (percent change)', 'output'), row('Private jobs (percent change)', 'jobs'),
           row('Labor services supplied (percent change)', 'supplied'),
           row('Output relative to capacity (percent)', 'use', 1, False),
           row('Labor services used (percent change)', 'used'), row('Wage bill (percent change)', 'wagebill'),
           row('Unemployment rate (percent)', 'unemp', 1, False),
           row('Residents with work (percent of residents)', 'anywork', 1, False)],
          'Changes are relative to the reference. Labor services are the efficiency units of the two occupations, added at their '
          'reference wages; those supplied are the services of employed workers, and those used are the share that producers buy, '
          'which is the rate at which capacity is used (Section~\\ref{sec:demand}). Residents with work hold a private job or are in '
          f"other employment; in the reference they are {f(main.start['anywork'], 1)} percent of residents, and the unemployment rate is "
          f"{f(main.start['unemp'], 1)} percent.")
    # the initial knowledge cohort: what the rise in capability changes (differences from the same cohort when
    # capability does not change; the two paths are no longer shown side by side, at the author's request, 18 November 2026)
    gap = main.cohort6 - main.cohort6_ref
    rows = [[str(years)] + [f(v, 1, True) for v in gap[q]] for years, q in ((2, 8), (6, 24), (10, 40), (20, 80), (40, LONG))]
    rows.append(['\\addlinespace Average, years 1--20'] + [f(v, 1, True) for v in gap[:FIRST].mean(0)])
    table('cohort', 'The initial knowledge cohort: what the rise in capability changes', 'tab:cohort',
          ['Years since the first rise in capability', '\\shortstack{Knowledge\\\\job}', '\\shortstack{Manual\\\\job}', '\\shortstack{Other\\\\employment}',
           'Unemployed', '\\shortstack{Inactive\\\\or retired}', '\\shortstack{Without\\\\work}'], rows,
          'Changes relative to the reference, in percentage points of the surviving members of the cohort of residents who are employed '
          'or unemployed in knowledge work at the initial date. '
          'Other employment is public employment and self-employment. The last column is the sum of the two '
          'columns before it: members without work are unemployed, inactive or retired. The last row is the average over the twenty '
          'years that follow the first rise in capability.',
          spec='lrrrrrr')
    qL = LONG
    without = lambda case: case.cohort[:, 2] - case.cohort_ref[:, 2] + case.cohort_out[:, 0] + case.cohort_out[:, 1]
    both = lambda label, value, d=1, sign=True: [label] + [f(at(value(case), q), d, sign) for case in (main, pol, tra) for q in ('avg', qL)]
    ser = lambda label, key, d=1, sign=True: both(label, lambda case: case.s[key], d, sign)
    jobs = lambda label, key: both(label, lambda case: 100 * (case.s[key] / case.start[key] - 1))
    table('experiments', 'Two policy experiments', 'tab:experiments',
          ['', '\\multicolumn{2}{c}{\\shortstack{No\\\\policy}}', '\\multicolumn{2}{c}{\\shortstack{Return follows\\\\capacity use}}',
           '\\multicolumn{2}{c}{\\shortstack{Transfer of\\\\licence rent}}'],
          [['Years'] + ['1--20', '40'] * 3,
           ['\\midrule ' + x if i == 0 else x for i, x in enumerate(ser('Output (percent change)', 'output'))],
           ser('Output relative to capacity (percent)', 'use', 1, False),
           ser('Return on claims (percent per year)', 'ret', 1, False),
           ser('Transfer to residents (percent of reference household consumption)', 'transfer', 1, False),
           ser('Household consumption (percent change)', 'cons'), ser('Capitalist consumption (percent change)', 'capcons'),
           ser('Conventional capital (percent change)', 'machines'),
           ser('Knowledge wage (percent change)', 'wageK'), ser('Manual wage (percent change)', 'wageM'),
           jobs('Knowledge jobs (percent change)', 'jobsK'), jobs('Manual jobs (percent change)', 'jobsM'),
           both('Residents outside the private workforce (million, change)', lambda case: case.s['outside'] - case.start['outside']),
           ser('Unemployment rate (percent)', 'unemp', 1, False),
           ['\\multicolumn{7}{@{}l}{Initial knowledge cohort (percentage points)}'],
           both('\\quad out of knowledge work', lambda case: case.cohort_ref[:, 0] - case.cohort[:, 0], 1, False),
           both('\\quad into manual work', lambda case: case.cohort[:, 1] - case.cohort_ref[:, 1], 1, False),
           both('\\quad without work', without, 1, False)],
          'The first two columns repeat the experiment of Section~\\ref{sec:numerics}, in which there is no policy. In the next two the '
          'return on claims follows \\eqref{eq:policy-rate} and can fall to zero (Section~\\ref{sec:policy}). In the last two, half '
          'of the rise in the licence rent is paid to residents (Section~\\ref{sec:transfer}). Columns headed 1--20 report averages over the twenty years that follow the first rise in '
          'capability; columns headed 40 report the outcome after forty years. Changes are relative to the reference; the last three rows '
          'are differences from the same cohort when capability does not change, and residents without work are unemployed, inactive or retired.',
          spec='l' + '@{\\hspace{1.4em}}'.join(['>{\\raggedleft\\arraybackslash}p{3.1em}' * 2] * 3),
          rules='\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(l){6-7}')
    # the limit case of the appendix: public purchases keep capacity in use
    both = lambda label, value, d=1, sign=True: [label] + [f(at(value(case), q), d, sign) for case in (main, pur) for q in ('avg', qL)]
    ser = lambda label, key, d=1, sign=True: both(label, lambda case: case.s[key], d, sign)
    jobs = lambda label, key: both(label, lambda case: 100 * (case.s[key] / case.start[key] - 1))
    table('limit', 'A limit case: public purchases keep capacity in use', 'tab:limit',
          ['', '\\multicolumn{2}{c}{No policy}', '\\multicolumn{2}{c}{Capacity kept in use}'],
          [['Years', '1--20', '40', '1--20', '40'],
           ['\\midrule ' + x if i == 0 else x for i, x in enumerate(ser('Output (percent change)', 'output'))],
           ser('Capacity (percent change)', 'capacity'),
           ser('Output relative to capacity (percent)', 'use', 1, False),
           ser('Public purchases (percent of output)', 'purshare', 1, False),
           ser('Household consumption (percent change)', 'cons'), ser('Capitalist consumption (percent change)', 'capcons'),
           ser('Conventional capital (percent change)', 'machines'),
           ser('AI capital (multiple of its level in the reference)', 'aicap', 1, False),
           ser('Knowledge wage (percent change)', 'wageK'), ser('Manual wage (percent change)', 'wageM'),
           jobs('Knowledge jobs (percent change)', 'jobsK'), jobs('Manual jobs (percent change)', 'jobsM'),
           both('Residents outside the private workforce (million, change)', lambda case: case.s['outside'] - case.start['outside']),
           ser('Unemployment rate (percent)', 'unemp', 1, False),
           ['\\multicolumn{5}{@{}l}{Initial knowledge cohort (percentage points)}'],
           both('\\quad out of knowledge work', lambda case: case.cohort_ref[:, 0] - case.cohort[:, 0], 1, False),
           both('\\quad into manual work', lambda case: case.cohort[:, 1] - case.cohort_ref[:, 1], 1, False),
           both('\\quad without work', without, 1, False)],
          'The first two columns repeat the experiment of Section~\\ref{sec:numerics}. In the last two the government buys final goods '
          'according to \\eqref{eq:purchases}. Columns headed 1--20 report averages over the twenty years that follow the first rise in '
          'capability; columns headed 40 report the outcome after forty years. Changes are relative to the reference; the last three rows '
          'are differences from the same cohort when capability does not change, and residents without work are unemployed, inactive or retired.',
          spec='l' + '@{\\hspace{1.8em}}'.join(['>{\\raggedleft\\arraybackslash}p{3.6em}' * 2] * 2),
          rules='\\cmidrule(lr){2-3}\\cmidrule(l){4-5}')
    both = lambda label, value, d=1, sign=True: [label] + [f(value(case, q), d, sign) for case in (main, anc) for q in (24, qL)]
    ser = lambda label, key, d=1, sign=True: both(label, lambda case, q: case.s[key][q], d, sign)
    jobs = lambda label, key: both(label, lambda case, q: 100 * (case.s[key][q] / case.start[key] - 1))
    table('anchor', 'The experiment when the consumption of capitalists follows their income', 'tab:anchor',
          ['', '\\multicolumn{2}{c}{Anchor fixed}', '\\multicolumn{2}{c}{Anchor follows income}'],
          [['', '6 years', '40 years', '6 years', '40 years'],
           ser('Capacity (percent change)', 'capacity'), ser('Output (percent change)', 'output'),
           ser('Output relative to capacity (percent)', 'use', 1, False),
           ser('Income of capitalists (percent change)', 'capincome'), ser('Capitalist consumption (percent change)', 'capcons'),
           ser('Household consumption (percent change)', 'cons'), ser('Wage bill (percent change)', 'wagebill'),
           ser('Knowledge wage (percent change)', 'wageK'), ser('Manual wage (percent change)', 'wageM'),
           jobs('Knowledge jobs (percent change)', 'jobsK'), jobs('Manual jobs (percent change)', 'jobsM'),
           both('Residents outside the private workforce (million, change)', lambda case, q: case.s['outside'][q] - case.start['outside']),
           both('Initial knowledge cohort: out of knowledge work (percentage points)', lambda case, q: case.cohort_ref[q, 0] - case.cohort[q, 0], 1, False),
           both('Initial knowledge cohort: into manual work (percentage points)', lambda case, q: case.cohort[q, 1] - case.cohort_ref[q, 1], 1, False),
           both('Initial knowledge cohort: without work (percentage points)', lambda case, q: case.cohort[q, 2] - case.cohort_ref[q, 2] + case.cohort_out[q, 0] + case.cohort_out[q, 1], 1, False)],
          'The first two columns repeat the experiment. In the last two the income at which the consumption of capitalists is '
          'anchored follows \\eqref{eq:moving-anchor}. Changes are relative to the reference; the last three rows are differences '
          'from the same cohort when capability does not change; residents without work are unemployed, inactive or retired.', spec='lrrrr')
    def variant(name, caption, label, heads, case, notes, first=None, extra=()):
        """The experiment beside one of its variants, six, ten and forty years after the first rise in capability."""
        both = lambda lab, value, d=1, sign=True: [lab] + [f(value(c, q), d, sign) for c in (main, case) for q in (24, 40, qL)]
        ser = lambda lab, key, d=1, sign=True: both(lab, lambda c, q: c.s[key][q], d, sign)
        jobs = lambda lab, key: both(lab, lambda c, q: 100 * (c.s[key][q] / c.start[key] - 1))
        rows = [['Years', '6', '10', '40', '6', '10', '40']] + ([ser(*first)] if first else []) + [
            ser('Output (percent change)', 'output'), ser('Output relative to capacity (percent)', 'use', 1, False),
            ser('Household consumption (percent change)', 'cons'), ser('Capitalist consumption (percent change)', 'capcons'),
            ser('Knowledge wage (percent change)', 'wageK'), ser('Manual wage (percent change)', 'wageM'),
            jobs('Knowledge jobs (percent change)', 'jobsK'), jobs('Manual jobs (percent change)', 'jobsM'),
            both('Residents outside the private workforce (million, change)', lambda c, q: c.s['outside'][q] - c.start['outside']),
            ser('Unemployment rate (percent)', 'unemp', 1, False)] + [ser(*x) for x in extra] + [
            both('Initial knowledge cohort: out of knowledge work (percentage points)', lambda c, q: c.cohort_ref[q, 0] - c.cohort[q, 0], 1, False),
            both('Initial knowledge cohort: into manual work (percentage points)', lambda c, q: c.cohort[q, 1] - c.cohort_ref[q, 1], 1, False),
            both('Initial knowledge cohort: without work (percentage points)',
                 lambda c, q: c.cohort[q, 2] - c.cohort_ref[q, 2] + c.cohort_out[q, 0] + c.cohort_out[q, 1], 1, False)]
        rows[1][0] = '\\midrule ' + rows[1][0]
        table(name, caption, label, [''] + ['\\multicolumn{3}{c}{%s}' % h for h in heads], rows,
              notes + ' Changes are relative to the reference; the last three rows are differences from the same cohort when capability '
              'does not change, and residents without work are unemployed, inactive or retired.', spec='lrrrrrr',
              rules='\\cmidrule(lr){2-4}\\cmidrule(l){5-7}')
    variant('spending', 'The experiment when capitalists consume more of an increase in their income', 'tab:spending',
            ['$c_C=0.5$', '$c_C=0.75$'], mpc,
            'The first three columns repeat the experiment. In the last three capitalists consume three quarters of an increase in their income.')
    variant('rentshare', 'The experiment when AI capital requires more produced goods', 'tab:rentshare',
            ['72 percent in the reference', '36 percent in the reference'], rent,
            'The column headings give the licence rent in the price of AI capital in the reference, at the normal use of capacity; '
            'the first row gives it along the path (Section~\\ref{sec:demand}). The first three columns repeat the experiment. In the '
            'last three, changes are relative to the reference computed with the lower rent.',
            first=('Licence rent in the price of AI capital (percent)', 'rentshare', 1, False))
    variant('ownership', 'The experiment when residents hold more of the claims on business payouts', 'tab:ownership',
            ['Residents hold 1 percent', 'Residents hold 30 percent'], own,
            'The first three columns repeat the experiment. In the last three residents hold 30 percent of the claims on business '
            'payouts, and changes are relative to the reference computed with that share.',
            first=('Payouts (percent change)', 'payouts'))
    variant('wagespeed', 'The experiment when posted wages adjust more slowly', 'tab:wagespeed',
            ['$\\varphi_g=0.25$', '$\\varphi_g=0.10$'], slow,
            'The first three columns repeat the experiment. In the last three posted wages close 10 percent of the gap to their '
            'flexible level each quarter. Labor services used are the efficiency units of employed workers that producers buy, added over the '
            'two occupations at reference wages.', first=('Job finding, knowledge work (per quarter)', 'findK', 2, False),
            extra=(('Private jobs (percent change)', 'jobs'), ('Labor services used (percent change)', 'used'), ('Wage bill (percent change)', 'wagebill')))
    rows = []
    names = (('Knowledge-home members, by qualification for manual work', None), ('\\quad not qualified', 'NoQual'),
             ('\\quad qualified', 'Qual'), ('Manual-home members', 'ManualHome'),
             ('Thirds of the cohort by initial liquid assets', None), ('\\quad lowest', 'LowSave'), ('\\quad middle', 'MidSave'),
             ('\\quad highest', 'HighSave'), ('All members', 'All'))
    for label, name in names:
        if name is None:
            rows.append([('\\midrule ' if not rows else '') + f'\\multicolumn{{12}}{{@{{}}l}}{{{label}}}']); continue
        idx = list(range(9)) if name == 'All' else Case.GROUPS[name]
        g = [main.group(idx, q) for q in (24, qL)]
        rows.append([label, f(g[0]['share'], 0)] + [f(x[k], d) for x in g for k, d in (('outK', 1), ('intoM', 1), ('other', 1), ('nowork', 1), ('cons', 0))])
    table('groups', 'The initial knowledge cohort by initial qualification and initial liquid assets', 'tab:groups',
          ['', '', '\\multicolumn{5}{c}{After 6 years}', '\\multicolumn{5}{c}{After 40 years}'],
          [['', 'Share', 'Out', 'Manual', 'Other', 'None', 'Cons.', 'Out', 'Manual', 'Other', 'None', 'Cons.']] + rows,
          'Members of the initial knowledge cohort grouped by their state at the initial date. ``Share\'\' is the percent of the cohort '
          'in the group. ``Out\'\' is the fall in the share of the group in a knowledge job, ``Manual\'\' the rise in the share in a manual '
          'job, ``Other\'\' the rise in the share in other employment (public employment and self-employment) and ``None\'\' the rise in the '
          'share without work (unemployed, inactive or retired), in percentage points of the surviving members of the group and relative to '
          'the reference. ``Cons.\'\' is the percent difference in consumption per surviving member relative to the reference. '
          'The thirds order all members of the cohort by their liquid assets at the initial date.', spec='lr@{\\hspace{0.9em}}rrrrr@{\\hspace{0.9em}}rrrrr', sep=3.0,
          rules='\\cmidrule(lr){3-7}\\cmidrule(l){8-12}')
    cases3 = (main, anc)
    both = lambda label, value, d=1, sign=True: [label] + [f(at(value(case), q), d, sign) for case in cases3 for q in ('avg', qL)]
    ser = lambda label, key, d=1, sign=True: both(label, lambda case: case.s[key], d, sign)
    jobs = lambda label, key: both(label, lambda case: 100 * (case.s[key] / case.start[key] - 1))
    num = '>{\\raggedleft\\arraybackslash}p{4.2em}'
    table('regimes', 'Two states of demand: aggregate outcomes and the initial knowledge cohort', 'tab:regimes',
          ['', '\\multicolumn{2}{c}{Anchor fixed}', '\\multicolumn{2}{c}{Anchor follows income}'],
          [['Years', '1--20', '40', '1--20', '40'],
           ['\\midrule ' + x if i == 0 else x for i, x in enumerate(ser('Output relative to capacity (percent)', 'use', 1, False))],
           ser('Output (percent change)', 'output'),
           ser('Household consumption (percent change)', 'cons'),
           ser('Knowledge wage (percent change)', 'wageK'), ser('Manual wage (percent change)', 'wageM'),
           jobs('Knowledge jobs (percent change)', 'jobsK'), jobs('Manual jobs (percent change)', 'jobsM'),
           ['\\multicolumn{5}{@{}l}{\\emph{Initial knowledge cohort (percentage points)}}'],
           both('\\quad out of knowledge work', lambda case: case.cohort_ref[:, 0] - case.cohort[:, 0], 1, False),
           both('\\quad into manual work', lambda case: case.cohort[:, 1] - case.cohort_ref[:, 1], 1, False),
           both('\\quad into other employment', lambda case: case.cohort_out[:, 2], 1, False),
           both('\\quad without work', without, 1, False)],
          'Columns headed 1--20 report averages over the twenty years that follow the first rise in capability; columns headed 40 report '
          'the outcome after forty years. The first two columns are the experiment of Section~\\ref{sec:numerics}, in which the '
          'consumption of capitalists is anchored at their income in the reference. In the last two the anchor follows \\eqref{eq:moving-anchor}. '
          'Technology, the path of capability, households, the labor market and the return on claims are the same in the two cases. Changes are relative to '
          'the reference. The cohort rows are differences from the same cohort when capability does not change; other employment is public '
          'employment and self-employment, and residents without work are unemployed, inactive or retired.',
          spec='l' + ('@{\\hspace{1.8em}}'.join([num * 2] * 2)),
          rules='\\cmidrule(lr){2-3}\\cmidrule(l){4-5}')
    cases6 = (main, acc, anc, ancacc)
    def block(q, short):
        both = lambda label, value, d=1, sign=True: [label] + [f(at(value(case), q), d, sign) for case in cases6]
        ser = lambda label, key, d=1, sign=True: both(label, lambda case: case.s[key], d, sign)
        jobs = lambda label, key: both(label, lambda case: 100 * (case.s[key] / case.start[key] - 1))
        rows = [] if short else [ser('Output (percent change)', 'output'), ser('Household consumption (percent change)', 'cons'),
                                 ser('Knowledge wage (percent change)', 'wageK')]
        rows += [ser('Manual wage (percent change)', 'wageM')]
        rows += [] if short else [jobs('Knowledge jobs (percent change)', 'jobsK')]
        rows += [jobs('Manual jobs (percent change)', 'jobsM')]
        rows += [] if short else [ser('Job finding, manual work (per quarter)', 'findM', 2, False)]
        rows += [both('Manual-home residents in a manual job (percent)', lambda case: case.home_manual()[0], 1, False),
                 ['\\multicolumn{5}{@{}l}{Initial knowledge cohort (percentage points)}'],
                 both('\\quad out of knowledge work', lambda case: case.cohort_ref[:, 0] - case.cohort[:, 0], 1, False),
                 both('\\quad into manual work', lambda case: case.cohort[:, 1] - case.cohort_ref[:, 1], 1, False)]
        rows += [] if short else [both('\\quad into other employment', lambda case: case.cohort_out[:, 2], 1, False)]
        rows += [both('\\quad without work', without, 1, False)]
        return rows
    def matched():
        """Capability measured against the path with the same access and constant capability (available for the experiment)."""
        def pct(label, key, name):
            own = lambda case: case.s[name] if name == 'wageM' else 100 * (case.s[name] / case.start[name] - 1)
            return [label, f(at(own(main), 'avg'), 1, True), f(at(100 * (acc.p[key][:, M] / ctl.p[key][:, M] - 1), 'avg'), 1, True),
                    f(at(own(anc), 'avg'), 1, True), f(at(100 * (ancacc.p[key][:, M] / ancacc.control.p[key][:, M] - 1), 'avg'), 1, True)]
        def pts(label, value):
            return [label, f(at(value(main), 'avg'), 1), f(at(value(acc) - value(ctl), 'avg'), 1), f(at(value(anc), 'avg'), 1), f(at(value(ancacc) - value(ancacc.control), 'avg'), 1)]
        return [pct('Manual wage (percent change)', 'wage', 'wageM'), pct('Manual jobs (percent change)', 'employment', 'jobsM'),
                ['\\multicolumn{5}{@{}l}{Initial knowledge cohort (percentage points)}'],
                pts('\\quad out of knowledge work', lambda case: case.cohort_ref[:, 0] - case.cohort[:, 0]),
                pts('\\quad into manual work', lambda case: case.cohort[:, 1] - case.cohort_ref[:, 1]),
                pts('\\quad into other employment', lambda case: case.cohort_out[:, 2]),
                pts('\\quad without work', without)]
    num = '>{\\raggedleft\\arraybackslash}p{4.9em}'
    table('access', 'Easier access to manual work under two states of demand', 'tab:access',
          ['', '\\multicolumn{2}{c}{Anchor fixed}', '\\multicolumn{2}{c}{Anchor follows income}'],
          [['Access to manual work', 'unchanged', 'easier', 'unchanged', 'easier'],
           ['\\midrule \\multicolumn{5}{@{}l}{\\emph{Average over years 1--20}}']] + block('avg', False)
          + [['\\addlinespace \\multicolumn{5}{@{}l}{\\emph{After 40 years}}']] + block(qL, True)
          + [['\\addlinespace \\multicolumn{5}{@{}l}{\\emph{Average over years 1--20, against the same access without a rise in capability}}']] + matched(),
          'With easier access every knowledge-home resident is qualified for manual work from the initial date, and so is every later '
          'entrant. The first panel reports averages over the twenty years that follow the first rise in capability. The columns with unchanged access repeat the experiment of Section~\\ref{sec:numerics} and the case of '
          'Section~\\ref{sec:anchor}. All changes are relative to the same reference, in which neither capability nor access changes; '
          f"there job finding in manual work is {f(main.start['findM'], 2)} per quarter and {f(main.home_manual()[1], 1)} percent of "
          'manual-home residents hold a manual job. The cohort rows are differences from the same cohort in that reference, so that with '
          'easier access they contain the effect of easier access as well as that of capability. In the third panel the outcomes with '
          'easier access are measured against a path on which access is easier and capability does not change, so that they contain the '
          'effect of capability alone; with unchanged access that path is the reference, and the columns repeat the first panel. Other employment is public employment '
          'and self-employment, and residents without work are unemployed, inactive or retired.',
          spec='l' + ('@{\\hspace{1.4em}}'.join([num * 2] * 2)),
          rules='\\cmidrule(lr){2-3}\\cmidrule(l){4-5}')
    # the short table of the robustness section: two spending rules and two rules of access, averages over years 1-20
    order = (main, anc, acc, ancacc)
    short = lambda label, value: [label] + [f(at(value(case), 'avg'), 1, True) for case in order]
    table('robust', 'The spending rule and access to manual work: averages over the first twenty years', 'tab:robust',
          ['', '\\multicolumn{2}{c}{Baseline access}', '\\multicolumn{2}{c}{Broader access}'],
          [['', '\\shortstack{Main\\\\experiment}', 'Benchmark', '\\shortstack{Main\\\\experiment}', 'Benchmark'],
           ['\\midrule Output (percent change)'] + short('', lambda case: case.s['output'])[1:],
           short('Manual wage (percent change)', lambda case: case.s['wageM']),
           ['\\multicolumn{5}{@{}l}{Initial knowledge cohort (percentage points)}'],
           short('\\quad in knowledge work', lambda case: case.cohort[:, 0] - case.cohort_ref[:, 0]),
           short('\\quad in manual work', lambda case: case.cohort[:, 1] - case.cohort_ref[:, 1]),
           short('\\quad without work', without)],
          'Averages over the twenty years that follow the first rise in AI capability. In the main experiment the consumption of '
          'capitalists is anchored at their income in the reference; in the benchmark the anchor follows \\eqref{eq:moving-anchor}. '
          'With broader access every knowledge-home resident is qualified for manual work from the initial date, and so is every later '
          'entrant. All changes are relative to the same reference, which has baseline access, so that the columns with broader access '
          'contain the effect of broader access itself. The cohort rows are changes in the shares of the surviving members of the '
          'initial knowledge cohort; members without work are unemployed, inactive or retired.',
          spec='l' + ('@{\\hspace{1.4em}}'.join([num * 2] * 2)),
          rules='\\cmidrule(lr){2-3}\\cmidrule(l){4-5}')
    cases5 = (main, anc, acc, own, pol, tra)
    line = lambda label, value: [label] + [f(value(case), 1, True) for case in cases5]
    head = ['', '\\shortstack{No\\\\policy}', '\\shortstack{Anchor\\\\follows\\\\income}', '\\shortstack{Easier\\\\access}',
            '\\shortstack{Residents\\\\hold 30\\\\percent}', '\\shortstack{Return\\\\follows\\\\capacity\\\\use}',
            '\\shortstack{Transfer\\\\of licence\\\\rent}']
    sections = ['Section', '\\ref{sec:numerics}', '\\ref{sec:anchor}', '\\ref{sec:access}', '\\ref{sec:ownership}', '\\ref{sec:policy}', '\\ref{sec:transfer}']
    well = lambda label, group, social=False: [label] + [f((case.social if social else case.welfare)(case.residents()[group]), 1, True) for case in cases5]
    groups = (('All residents', 'All'), ('Initial knowledge cohort', 'Cohort'), ('\\quad not qualified', 'NoQual'), ('\\quad qualified', 'Qual'),
              ('\\quad manual-home members', 'ManualHome'), ('Manual workers', 'Manual'), ('\\quad manual-home', 'ManualM'),
              ('\\quad knowledge-home', 'ManualK'), ('Inactive residents', 'Inactive'),
              ('Retired residents', 'Retired'), ('Other employment', 'OtherWork'))
    table('welfare', 'The welfare of residents alive at the initial date', 'tab:welfare', head,
          [sections, ['\\midrule \\multicolumn{7}{@{}l}{\\emph{Dates discounted as residents discount them}}']]
          + [well(label, group) for label, group in groups]
          + [['\\addlinespace \\multicolumn{7}{@{}l}{\\emph{Dates discounted at the return on claims}}']]
          + [well(label, group, True) for label, group in groups],
          'Consumption-equivalent change in welfare \\eqref{eq:welfare}, in percent, of groups of the residents who are alive at the initial '
          'date, by their state at that date: the change of consumption at every date and in every state of the reference that gives the '
          'group, on average, the value that the path gives it. In the first panel dates are weighted with the discount factors of the '
          'residents; in the second their choices are the same and every date is discounted at the return on claims in the reference. '
          'The initial knowledge cohort is divided into knowledge-home members who are not qualified and who are qualified for manual work, '
          'and manual-home members. Manual workers are employed or unemployed in manual work at the initial date; they are divided into manual-home residents and knowledge-home residents who hold or seek manual jobs. Residents who enter later '
          'are in Table~\\ref{tab:later}, and capitalists have no measure of welfare. In the second column the anchor of the consumption of '
          'capitalists follows their income. With easier access the changes contain the effect of easier access itself as well as that of '
          'capability; Table~\\ref{tab:access-welfare} separates the two. When residents hold 30 percent of the claims on business payouts, the reference is the one computed with that share. '
          'When the return follows the use of capacity, capacity is used above its normal rate for six years, at no cost to residents.',
          spec='l' + '>{\\raggedleft\\arraybackslash}p{4.6em}' * 6, sep=4.0)
    # the main experiment alone, for Section 6.5 (18 November 2026): groups of residents and later entrants, two ways of discounting
    alone = lambda label, group: [label, f(main.welfare(main.residents()[group]), 1, True), f(main.social(main.residents()[group]), 1, True)]
    entry = lambda label, q: [label, f(main.entrant(q, False), 1, True), f(main.entrant(q, True), 1, True)]
    table('welfaremain', 'Welfare in the main experiment', 'tab:welfare-main',
          ['', 'Own discount factors', 'Reference interest rate'],
          [['\\multicolumn{3}{@{}l}{\\emph{Residents alive at the initial date}}']] + [alone(label, group) for label, group in groups]
          + [['\\addlinespace \\multicolumn{3}{@{}l}{\\emph{Residents who enter later}}'], entry('\\quad after 10 years', 40),
             entry('\\quad after 20 years', 80), entry('\\quad after 40 years', LONG)],
          'Consumption-equivalent change in welfare \\eqref{eq:welfare}, in percent: the change of consumption at every date and in every '
          'state of the reference that gives a group, on average, the value that it has in the main experiment. In the first column dates '
          'are weighted with the discount factors of the residents; in the second their choices are the same and every date is discounted '
          'at the reference interest rate. Groups are defined by the state of residents at the initial date. The initial knowledge cohort '
          'is divided into knowledge-home members who are not qualified and who are qualified for manual work, and manual-home members. '
          'Manual workers hold or seek manual jobs at the initial date; they are divided by home occupation. Residents who enter later are '
          'measured relative to residents who enter in the reference. Capitalists have no measure of welfare.',
          spec='l' + '>{\\raggedleft\\arraybackslash}p{9em}' * 2)
    def four(label, group, social):
        fn = lambda case: (case.social if social else case.welfare)(case.residents()[group])
        alone, joint = fn(ctl), fn(acc)
        return [label, f(alone, 1, True), f(fn(main), 1, True), f(100 * ((1 + joint / 100) / (1 + alone / 100) - 1), 1, True), f(joint, 1, True)]
    rows6 = (('All residents', 'All'), ('Initial knowledge cohort', 'Cohort'), ('\\quad not qualified', 'NoQual'), ('\\quad qualified', 'Qual'),
             ('Manual workers', 'Manual'), ('\\quad manual-home', 'ManualM'), ('\\quad knowledge-home', 'ManualK'), ('Inactive residents', 'Inactive'))
    table('accesswelfare', 'Easier access and the welfare of residents alive at the initial date', 'tab:access-welfare',
          ['', '\\shortstack{Easier access,\\\\no rise in\\\\capability}', '\\shortstack{Capability,\\\\unchanged\\\\access}',
           '\\shortstack{Capability,\\\\easier\\\\access}', '\\shortstack{Easier access\\\\and\\\\capability}'],
          [['\\multicolumn{5}{@{}l}{\\emph{Dates discounted as residents discount them}}']] + [four(label, group, False) for label, group in rows6]
          + [['\\addlinespace \\multicolumn{5}{@{}l}{\\emph{Dates discounted at the return on claims}}']] + [four(label, group, True) for label, group in rows6],
          'Consumption-equivalent changes in welfare \\eqref{eq:welfare}, in percent, for the experiment of Section~\\ref{sec:numerics}. The first column '
          'compares the path on which access is easier and capability does not change with the reference. The second repeats the '
          'experiment with unchanged access. The last compares the path with easier access and rising capability with the reference. '
          'The third is the change from the first path to the last, computed as the ratio of the two consumption equivalents, so that it '
          'measures the effect of capability when access is easier. Groups are defined by the state of residents at the initial date, '
          'as in Table~\\ref{tab:welfare}; qualification refers to manual work before access is made easier.',
          spec='l' + '>{\\raggedleft\\arraybackslash}p{6.2em}' * 4, sep=4.0)
    enter = lambda label, q, social=False: [label] + [f(case.entrant(q, social), 1, True) for case in cases5]
    table('later', 'Residents who enter later, and present values', 'tab:later', head,
          [sections, ['\\midrule \\multicolumn{7}{@{}l}{\\emph{Welfare of a resident who enters later, dates discounted as residents discount them}}'],
           enter('\\quad after 10 years', 40), enter('\\quad after 20 years', 80), enter('\\quad after 40 years', LONG),
           ['\\addlinespace \\multicolumn{7}{@{}l}{\\emph{Welfare of a resident who enters later, dates discounted at the return on claims}}'],
           enter('\\quad after 10 years', 40, True), enter('\\quad after 20 years', 80, True), enter('\\quad after 40 years', LONG, True),
           ['\\addlinespace \\multicolumn{7}{@{}l}{\\emph{Present value over forty years (percent difference from the reference)}}'],
           line('Consumption of the cohort', lambda case: consumption_loss(case)),
           line('Household consumption', lambda case: present_value(case, case.p['household_consumption'], case.acc['household_consumption'])),
           line('Capitalist consumption', lambda case: present_value(case, case.p['capitalist_consumption'], case.acc['capitalist_consumption'])),
           line('Wage bill', lambda case: present_value(case, case.p['payroll'], case.ag['payroll'])),
           line('Output', lambda case: present_value(case, case.p['output'], case.acc['output']))],
          'The first two panels report the consumption-equivalent change in welfare \\eqref{eq:welfare}, in percent, of a resident who '
          'enters the economy ten, twenty or forty years after the first rise in capability, against a resident who enters in the reference. '
          'The entrant has the type, the status and the liquid claims of the residents who enter in a quarter of the reference, most of whom '
          'are inactive and hold no liquid claims. The third panel reports the percent difference from the reference in the present value of '
          'each quantity over the forty years that follow the first rise in capability, discounted at the return on claims in the reference; '
          'for the cohort the quantity is the consumption of its surviving members. The columns are those of Table~\\ref{tab:welfare}.',
          spec='l' + '>{\\raggedleft\\arraybackslash}p{4.6em}' * 6, sep=4.0)
    four4 = [boom(case) for case in (pol,) + tuple(tests)]
    line4 = lambda label, key, d=1, sign=False: [label] + [f(b[key], d, sign) for b in four4]
    table('boom', 'The boom when the return follows the use of capacity: three tests', 'tab:boom',
          ['', '\\shortstack{Experiment of\\\\Section~\\ref{sec:policy}}', '\\shortstack{Capability\\\\foreseen}', '$c_C=0.25$', '$c_C=0.75$'],
          [line4('Peak of output (percent above the reference)', 'peak'), line4('\\quad year of the peak', 'year', 0),
           ['\\multicolumn{5}{@{}l}{Spending at the peak (points of output in the reference, change)}'],
           line4('\\quad investment in conventional capital', 'invest', 1, True), line4('\\quad consumption of capitalists', 'capcons', 1, True),
           line4('\\quad household consumption', 'households', 1, True), line4('\\quad recruitment and installation', 'other', 1, True),
           line4('Highest use of capacity (percent)', 'usemax'),
           line4('Output falls short of capacity from year', 'slack', 0), line4('Return on claims at zero from year', 'zero', 0),
           line4('Highest return on claims (percent per year)', 'retmax'),
           ['\\multicolumn{5}{@{}l}{Initial knowledge cohort without work (percentage points)}'],
           line4('\\quad average over years 1--20', 'nowork'), line4('\\quad largest', 'worst'), line4('\\quad year in which it is largest', 'worstyear', 0),
           line4('Output after 40 years (percent change)', 'output', 1, True), line4('Household consumption after 40 years (percent change)', 'cons', 1, True)],
          'The first column repeats the experiment of Section~\\ref{sec:policy}, in which the return on claims follows \\eqref{eq:policy-rate}, '
          'capability rises in yearly steps that are not foreseen and capitalists consume half of an increase in their income, $c_C=0.5$. '
          'In the second column the whole path of the frontier, which rises evenly from '
          f"{f(main.start['ai'], 0)} to {f(main.s['frontier'][qL], 0)} percent over six years, is known from the initial date. In the last two columns "
          'capitalists consume a quarter and three quarters of an increase in their income. Spending at the peak is the change from the '
          'reference in the quarter in which output is highest; investment in conventional capital includes the capital with which AI capital is supplied. '
          'Members of the cohort without work are unemployed, inactive or retired, and are measured as differences from the same cohort when capability does not change.',
          spec='lrrrr')
    napp, nacc, nctl = main.noapp                           # without applications of employed knowledge workers for manual jobs
    six = (main, napp, ctl, nctl, acc, nacc)
    wg = lambda c: c.cohort[:, 2] - c.cohort_ref[:, 2] + c.cohort_out[:, 0] + c.cohort_out[:, 1]
    avg6 = lambda label, fn, d=1, sign=True: [label] + [f(float(np.mean(fn(c)[:FIRST])), d, sign) for c in six]
    end6 = lambda label, fn, d=1, sign=True: [label] + [f(float(fn(c)[qL]), d, sign) for c in six]
    num6 = '>{\\raggedleft\\arraybackslash}p{3.6em}'
    table('applications', 'The experiment without applications of employed knowledge workers for manual jobs', 'tab:applications',
          ['', '\\multicolumn{2}{c}{\\shortstack{Capability,\\\\unchanged access}}', '\\multicolumn{2}{c}{\\shortstack{Easier access,\\\\no rise in capability}}',
           '\\multicolumn{2}{c}{\\shortstack{Capability,\\\\easier access}}'],
          [['Applications of employed knowledge workers', 'with', 'without', 'with', 'without', 'with', 'without'],
           ['\\midrule \\multicolumn{7}{@{}l}{\\emph{Average over years 1--20}}'],
           avg6('Job finding, manual work (per quarter)', lambda c: c.s['findM'], 2, False),
           avg6('Manual wage (percent change)', lambda c: c.s['wageM']),
           avg6('Manual jobs (percent change)', lambda c: 100 * (c.s['jobsM'] / c.start['jobsM'] - 1)),
           avg6('Labor services supplied in manual work (percent change)', lambda c: 100 * (c.p['labour'][:, M] / c.ag['labour'][M] - 1)),
           avg6('Manual-home residents in a manual job (percent)', lambda c: c.home_manual()[0], 1, False),
           avg6('Output (percent change)', lambda c: c.s['output']),
           ['\\multicolumn{7}{@{}l}{Initial knowledge cohort (percentage points)}'],
           avg6('\\quad out of knowledge work', lambda c: c.cohort_ref[:, 0] - c.cohort[:, 0], 1, False),
           avg6('\\quad into manual work', lambda c: c.cohort[:, 1] - c.cohort_ref[:, 1], 1, False),
           avg6('\\quad without work', wg, 1, False),
           ['\\addlinespace \\multicolumn{7}{@{}l}{\\emph{After 40 years}}'],
           end6('Manual wage (percent change)', lambda c: c.s['wageM']),
           end6('Manual jobs (percent change)', lambda c: 100 * (c.s['jobsM'] / c.start['jobsM'] - 1)),
           end6('Initial knowledge cohort without work (percentage points)', wg, 1, False)],
          'In the columns headed ``with\'\' employed knowledge workers who are qualified for manual work apply for manual jobs with the intensity of the '
          f"calibration, {f(main.eco.P.offer_intensity[N], 2)}; these columns repeat the experiment of Section~\\ref{{sec:numerics}} and the two paths with easier access of "
          'Section~\\ref{sec:access}. In the columns headed ``without\'\' the intensity is zero, and the reference is computed again; it reproduces the same data. '
          'All changes are relative to the reference, in which job finding in manual work is '
          f"{f(main.start['findM'], 2)} per quarter and {f(main.home_manual()[1], 1)} percent of manual-home residents hold a manual job. "
          'The cohort rows are differences from the same cohort in the reference; residents without work are unemployed, inactive or retired.',
          spec='l' + ('@{\\hspace{1.0em}}'.join([num6 * 2] * 3)), rules='\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(l){6-7}')
    quad = (main.rholow, (main, acc, ctl), main.rhohigh, (main.acclow, None, None))
    cell = lambda fn, which, d, sign: [('--' if t[which] is None else f(fn(t[which]), d, sign)) for t in quad]
    a4 = lambda label, fn, which=0, d=1, sign=True: [label] + cell(lambda c: float(np.mean(fn(c)[:FIRST])), which, d, sign)
    e4 = lambda label, fn, which=0, d=1, sign=True: [label] + cell(lambda c: float(fn(c)[qL]), which, d, sign)
    r4 = lambda label, key, d=1: [label] + cell(lambda c: run.fit(c.const, c.ref)[key], 0, d, False)
    num4 = '>{\\raggedleft\\arraybackslash}p{4.6em}'
    table('parameters', 'Earnings of knowledge workers in manual work, and the share qualified without instruction', 'tab:entry-checks',
          ['', '\\multicolumn{3}{c}{\\shortstack{Earnings in manual work\\\\relative to home earnings}}', '\\shortstack{A quarter\\\\qualified}'],
          [['', '0.80', '0.90', '1.00', ''],
           ['\\midrule \\multicolumn{5}{@{}l}{\\emph{The reference}}'],
           r4(f"Knowledge share of private employment (percent; data {f(par.TARGETS['knowledge employment share, %'], 1)})", 'knowledge employment share, %'),
           r4(f"Unemployed knowledge workers taking manual jobs (percent per quarter; data {f(par.TARGETS['N to M, unemployed origin, % per quarter'], 1)})", 'N to M, unemployed origin, % per quarter'),
           r4(f"Employed knowledge workers moving to manual jobs (percent per quarter; data {f(par.TARGETS['N to M, employed origin, % per quarter'], 2)})", 'N to M, employed origin, % per quarter', 2),
           ['\\addlinespace \\multicolumn{5}{@{}l}{\\emph{Capability, unchanged access: average over years 1--20}}'],
           a4('Job finding, manual work (per quarter)', lambda c: c.s['findM'], 0, 2, False),
           a4('Manual wage (percent change)', lambda c: c.s['wageM']),
           a4('Manual jobs (percent change)', lambda c: 100 * (c.s['jobsM'] / c.start['jobsM'] - 1)),
           a4('Labor services supplied in manual work (percent change)', lambda c: 100 * (c.p['labour'][:, M] / c.ag['labour'][M] - 1)),
           a4('Output (percent change)', lambda c: c.s['output']),
           ['\\multicolumn{5}{@{}l}{Initial knowledge cohort (percentage points)}'],
           a4('\\quad out of knowledge work', lambda c: c.cohort_ref[:, 0] - c.cohort[:, 0], 0, 1, False),
           a4('\\quad into manual work', lambda c: c.cohort[:, 1] - c.cohort_ref[:, 1], 0, 1, False),
           a4('\\quad without work', wg, 0, 1, False),
           ['\\addlinespace \\multicolumn{5}{@{}l}{\\emph{Capability, unchanged access: after 40 years}}'],
           e4('Manual wage (percent change)', lambda c: c.s['wageM']),
           e4('Initial knowledge cohort without work (percentage points)', wg, 0, 1, False),
           ['\\addlinespace \\multicolumn{5}{@{}l}{\\emph{Easier access, no rise in capability: average over years 1--20}}'],
           a4('Manual wage (percent change)', lambda c: c.s['wageM'], 2),
           a4('Job finding, manual work (per quarter)', lambda c: c.s['findM'], 2, 2, False),
           a4('Manual-home residents in a manual job (percent)', lambda c: c.home_manual()[0], 2, 1, False),
           ['\\addlinespace \\multicolumn{5}{@{}l}{\\emph{Capability, easier access: average over years 1--20}}'],
           a4('Manual wage (percent change)', lambda c: c.s['wageM'], 1),
           a4('Manual-home residents in a manual job (percent)', lambda c: c.home_manual()[0], 1, 1, False),
           a4('Initial knowledge cohort without work (percentage points)', wg, 1, 1, False)],
          'The column headed 0.90 repeats the experiment of Section~\\ref{sec:numerics} and the two paths with easier access of Section~\\ref{sec:access}. '
          'In the columns headed 0.80 and 1.00 a knowledge-home resident earns that multiple of home earnings in manual work at reference wages, and supplies labor services in manual work in the same proportion. '
          'In the last column a quarter, in place of a half, of the knowledge-home residents are qualified for manual work without instruction, and the ratio is 0.90; the paths with easier access are not computed for it. '
          'In each column the reference is computed again, and the first panel compares it with the data. All changes are relative to the reference of the column, in which '
          f"{f(main.home_manual()[1], 1)} percent of manual-home residents hold a manual job. "
          'The cohort rows are differences from the same cohort in the reference; residents without work are unemployed, inactive or retired.',
          spec='l' + num4 * 3 + '@{\\hspace{1.2em}}' + num4, rules='\\cmidrule(lr){2-4}\\cmidrule(l){5-5}')
    five = (main,) + tuple(checks[:2]) + (tra, checks[2])
    avg5 = lambda label, fn, d=1, sign=True: [label] + [f(float(np.mean(fn(c)[:FIRST])), d, sign) for c in five]
    end5 = lambda label, fn, d=1, sign=True: [label] + [f(float(fn(c)[qL]), d, sign) for c in five]
    one5 = lambda label, fn, d=1, sign=True: [label] + [f(float(fn(c)), d, sign) for c in five]
    table('checks', 'Further checks', 'tab:checks',
          ['', '\\shortstack{Experiment}', '\\shortstack{$\\eta=1$}', '\\shortstack{AI costs\\\\30 percent\\\\less}',
           '\\shortstack{Transfer of\\\\half of\\\\the rise}', '\\shortstack{Transfer of\\\\the whole\\\\rise}'],
          [['\\multicolumn{6}{@{}l}{\\emph{Average over years 1--20}}'],
           avg5('Output (percent change)', lambda c: c.s['output']), avg5('Output relative to capacity (percent)', lambda c: c.s['use'], 1, False),
           avg5('Knowledge wage (percent change)', lambda c: c.s['wageK']), avg5('Manual wage (percent change)', lambda c: c.s['wageM']),
           ['\\multicolumn{6}{@{}l}{Initial knowledge cohort (percentage points)}'],
           avg5('\\quad out of knowledge work', lambda c: c.cohort_ref[:, 0] - c.cohort[:, 0], 1, False),
           avg5('\\quad into manual work', lambda c: c.cohort[:, 1] - c.cohort_ref[:, 1], 1, False),
           avg5('\\quad without work', without, 1, False),
           ['\\addlinespace \\multicolumn{6}{@{}l}{\\emph{After 40 years}}'],
           end5('Output (percent change)', lambda c: c.s['output']), end5('Household consumption (percent change)', lambda c: c.s['cons']),
           end5('Knowledge wage (percent change)', lambda c: c.s['wageK']), end5('Manual wage (percent change)', lambda c: c.s['wageM']),
           end5('Transfer (percent of reference household consumption)', lambda c: c.s['transfer'], 1, False),
           ['\\addlinespace \\multicolumn{6}{@{}l}{\\emph{Welfare of residents alive at the initial date (consumption equivalent, percent)}}'],
           one5('Initial knowledge cohort, own discounting', lambda c: c.welfare(c.residents()['Cohort'])),
           one5('All residents, own discounting', lambda c: c.welfare(c.residents()['All'])),
           one5('Initial knowledge cohort, at the return on claims', lambda c: c.social(c.residents()['Cohort'])),
           one5('All residents, at the return on claims', lambda c: c.social(c.residents()['All']))],
          'The first column repeats the experiment of Section~\\ref{sec:numerics}. In the second the elasticity of substitution between '
          'knowledge and manual tasks is one instead of one half. In the third AI costs 30 percent less than knowledge labor on feasible tasks '
          'at reference prices, instead of 50 percent. In both, the reference is computed again and reproduces the same data, and changes are '
          'measured from it. The fourth column repeats the transfer of Section~\\ref{sec:transfer}, and in the fifth the whole rise of the '
          'licence rent is paid to residents. The cohort rows are differences from the same cohort when capability does not change; residents '
          'without work are unemployed, inactive or retired. Welfare is measured as in Table~\\ref{tab:welfare}.',
          spec='l' + '>{\\raggedleft\\arraybackslash}p{5.2em}' * 5, sep=4.0)
    fit = run.fit(main.const, main.ref)
    names = {'private employment, million': 'Private employment (million)',
             'knowledge employment share, %': 'Knowledge share of private employment (percent)',
             'private unemployment rate, %': 'Private unemployment rate (percent)',
             'inactive, million': 'Inactive residents (million)', 'retired, million': 'Retired residents (million)',
             'other employed, million': 'Other employed residents (million)',
             'N to M, unemployed origin, % per quarter': 'Unemployed knowledge workers taking manual jobs (percent per quarter)',
             'M to N, unemployed origin, % per quarter': 'Unemployed manual workers taking knowledge jobs (percent per quarter)',
             'N to M, employed origin, % per quarter': 'Employed knowledge workers moving to manual jobs (percent per quarter)',
             'M to N, employed origin, % per quarter': 'Employed manual workers moving to knowledge jobs (percent per quarter)',
             'zero liquid assets, all residents, %': 'Residents without liquid assets (percent)',
             'liquid assets, model units': 'Liquid assets (quarters of GDP)'}
    rows = []
    for k, label in names.items():
        scale = 1 / main.acc['gdp'] if k.startswith('liquid') else 1
        rows.append([label, f(fit[k] * scale, 2 if ', employed origin' in k or k.startswith('liquid') else 1),
                     f(par.TARGETS[k] * scale, 2 if ', employed origin' in k or k.startswith('liquid') else 1)])
    table('fit', 'The reference and the data', 'tab:fit', ['', 'Model', 'Data'], rows,
          'Data moments are those of the household calibration described in the text and in Appendix~\\ref{sec:data}. Liquid assets are divided by '
          'quarterly value added of the model in both columns. The four flows are not the same object in the two columns. In the model a flow is '
          'the share of the residents in the origin status at the start of a quarter who hold a job in the other occupation when income is paid '
          'in that quarter. In the data it is a monthly rate $r$ converted as $1-(1-r)^3$, the cumulative probability over a quarter under a '
          'constant monthly hazard. With other destinations and with returns, this can differ from the probability of being in the other '
          'occupation three months later, which has not been estimated directly from the linked records.', spec='lrr')


def reword(fig):
    """Put every text of a figure into the vocabulary of the paper."""
    import matplotlib.text as mtext
    for t in fig.findobj(mtext.Text):
        t.set_text(plain(t.get_text()))


def figures(main, anc, pol, tra, pur):
    import matplotlib
    matplotlib.use('pdf')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.titlesize': 9.5, 'axes.titleweight': 'bold',
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#52514e',
                         'axes.linewidth': 0.6, 'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.6,
                         'xtick.color': '#52514e', 'ytick.color': '#52514e', 'legend.frameon': False, 'legend.fontsize': 8.5,
                         'lines.linewidth': 1.6, 'pdf.fonttype': 42})
    H = 80                                                   # twenty years
    years = np.arange(H) / 4

    def panel(ax, title, series, ylabel, zero=False, loc='best', legend=True):
        for y, label, color, style in series:
            ax.plot(years, y[:H], color=color, ls=style, label=label)
        ax.set_title(title, loc='left'); ax.set_ylabel(ylabel); ax.set_xlim(0, 20); ax.set_xticks([0, 5, 10, 15, 20])
        if zero:
            ax.axhline(0, color='#52514e', lw=0.6)
        if len(series) > 1 and legend:
            ax.legend(loc=loc, handlelength=2.2)

    m = main.s
    flat = lambda v: np.full(main.T, v)
    fig, ax = plt.subplots(2, 2, figsize=(6.6, 5.0))
    panel(ax[0, 0], 'A. Knowledge tasks performed by AI', [(m['frontier'], 'Capability frontier', GREY, '--'), (m['ai'], 'Performed by AI', BLUE, '-')], 'Percent of tasks', loc='lower right')
    panel(ax[0, 1], 'B. Capacity and output', [(m['capacity'], 'Capacity', GREY, '--'), (m['output'], 'Output', BLUE, '-')], 'Percent change', zero=True, loc='center right')
    panel(ax[1, 0], 'C. Consumption', [(m['cons'], 'Households', BLUE, '-'), (m['capcons'], 'Capitalists', ORANGE, '-.')], 'Percent change', zero=True, loc='center right')
    panel(ax[1, 1], 'D. Posted wages', [(m['wageK'], 'Knowledge work', BLUE, '-'), (m['wageM'], 'Manual work', ORANGE, '-.')], 'Percent change', zero=True, loc='upper right')
    for a in ax[1]:
        a.set_xlabel('Years since the first rise in capability')
    reword(fig); fig.tight_layout(); fig.savefig(FIG / 'aggregates.pdf'); plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(6.6, 5.0))
    panel(ax[0, 0], 'A. Private jobs', [(m['jobsK'] - main.start['jobsK'], 'Knowledge work', BLUE, '-'), (m['jobsM'] - main.start['jobsM'], 'Manual work', ORANGE, '-.')], 'Million, change', zero=True, loc='center right')
    panel(ax[0, 1], 'B. Job finding', [(m['findK'], 'Knowledge work', BLUE, '-'), (m['findM'], 'Manual work', ORANGE, '-.')], 'Probability per quarter', loc='lower right')
    panel(ax[1, 0], 'C. Unemployment rate', [(m['unemp'], '', BLUE, '-')], 'Percent')
    panel(ax[1, 1], 'D. Outside the private workforce', [(m['outside'] - main.start['outside'], '', BLUE, '-')], 'Million, change', zero=True)
    ax[0, 1].set_ylim(0, None)
    for a in ax[1]:
        a.set_xlabel('Years since the first rise in capability')
    reword(fig); fig.tight_layout(); fig.savefig(FIG / 'labor.pdf'); plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(6.6, 5.0))
    one = lambda y: [(y, '', BLUE, '-')]                       # the effect of capability: differences from the same cohort
    panel(ax[0, 0], 'A. In a knowledge job', one(main.cohort[:, 0] - main.cohort_ref[:, 0]), 'Percentage points', zero=True)
    panel(ax[0, 1], 'B. In a manual job', one(main.cohort[:, 1] - main.cohort_ref[:, 1]), 'Percentage points', zero=True)
    panel(ax[1, 0], 'C. Without work', one(main.cohort6[:, 5] - main.cohort6_ref[:, 5]), 'Percentage points', zero=True)
    panel(ax[1, 1], 'D. Qualified for manual work', one(m['qualified']), 'Percent')
    for a in ax[1]:
        a.set_xlabel('Years since the first rise in capability')
    reword(fig); fig.tight_layout(); fig.savefig(FIG / 'cohort.pdf'); plt.close(fig)

    # the two states of demand over the twenty years that the comparison averages
    without = lambda c: c.cohort[:, 2] - c.cohort_ref[:, 2] + c.cohort_out[:, 0] + c.cohort_out[:, 1]
    three = lambda value: [(value(main), 'Anchor fixed (the experiment)', BLUE, '-'), (value(anc), 'Anchor follows income', ORANGE, '-.')]
    fig, ax = plt.subplots(2, 2, figsize=(6.6, 5.2))
    panel(ax[0, 0], 'A. Cohort without work', three(without), 'Percentage points', zero=True, legend=False)
    panel(ax[0, 1], 'B. Cohort in manual work', three(lambda c: c.cohort[:, 1] - c.cohort_ref[:, 1]), 'Percentage points', zero=True, legend=False)
    panel(ax[1, 0], 'C. Manual wage', three(lambda c: c.s['wageM']), 'Percent change', zero=True, legend=False)
    panel(ax[1, 1], 'D. Output', three(lambda c: c.s['output']), 'Percent change', zero=True, legend=False)
    for a in ax[1]:
        a.set_xlabel('Years since the first rise in capability')
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc='upper center', ncol=2, handlelength=2.2)
    reword(fig); fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(FIG / 'regimes.pdf'); plt.close(fig)

    # the experiment without policy and the three policy experiments
    four = lambda value: [(value(main), 'No policy', GREY, '--'), (value(pol), 'Return follows capacity use', BLUE, '-'),
                          (value(tra), 'Transfer of licence rent', ORANGE, '-.')]
    fig, ax = plt.subplots(2, 2, figsize=(6.6, 5.4))
    panel(ax[0, 0], 'A. Cohort without work', four(without), 'Percentage points', zero=True, legend=False)
    panel(ax[0, 1], 'B. Manual wage', four(lambda c: c.s['wageM']), 'Percent change', zero=True, legend=False)
    panel(ax[1, 0], 'C. Household consumption', four(lambda c: c.s['cons']), 'Percent change', zero=True, legend=False)
    panel(ax[1, 1], 'D. Output', four(lambda c: c.s['output']), 'Percent change', zero=True, legend=False)
    for a in ax[1]:
        a.set_xlabel('Years since the first rise in capability')
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc='upper center', ncol=3, handlelength=2.4, columnspacing=1.4)
    reword(fig); fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(FIG / 'experiments.pdf'); plt.close(fig)


def main():
    GEN.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    case = Case('self-financed')                             # the experiment: the return on claims is fixed
    pol = Case('self-financed', tag='_cut_main')             # policy experiment: the return follows the use of capacity and can fall to zero
    anc = Case('self-financed', tag='_anchor05')             # the anchor of capitalists' consumption follows their income
    mpc = Case('self-financed', tag='_mpc75')
    rent = Case('self-financed', tag='_rent36', rent_share=0.36)
    own = Case('self-financed', tag='_own30', payout_share=0.30)
    slow = Case('self-financed', tag='_wage10')
    acc, ancacc = (Case('self-financed', tag=t) for t in ('_access', '_access_anchor05'))
    ctl = Case('self-financed', tag='_access_noai')          # easier access without a rise in capability (the control)
    ancacc.control = Case('self-financed', tag='_access_anchor05_noai')     # the same control when the anchor follows income
    tra = Case('self-financed', tag='_transfer50')           # policy experiment: half of the rise in the licence rent goes to residents
    pur = Case('self-financed', tag='_purchases')            # policy experiment: public purchases keep capacity in use
    # tests of the boom when the return follows the use of capacity (Appendix D): capability foreseen, and capitalists who
    # consume a quarter and three quarters of an increase in their income
    tests = tuple(Case('self-financed', tag=t) for t in ('_cut_foresight', '_cut_mpc25', '_cut_mpc75'))
    # further checks (Appendix E): tasks neither complements nor substitutes, a smaller cost advantage of AI, and the
    # whole rise of the licence rent transferred
    checks = (Case('self-financed', tag='_eta10', eta=1.0), Case('self-financed', tag='_ai30', discount=0.30),
              Case('self-financed', tag='_transfer100'))
    # a check (Appendix E): employed knowledge workers do not apply for manual jobs; the experiment, easier access and its control
    case.noapp = tuple(Case('self-financed', tag=t, offer_knowledge=0.0) for t in ('_noapp', '_noapp_access', '_noapp_access_noai'))
    # checks (Appendix E): other earnings of knowledge-home residents in manual work (0.80 and 1.00 of home earnings in place
    # of 0.90), each with easier access and its control; and a quarter in place of a half of them qualified without instruction
    case.rholow = tuple(Case('self-financed', tag=t, rho_knowledge=0.8) for t in ('_rho80', '_rho80_access', '_rho80_access_noai'))
    case.rhohigh = tuple(Case('self-financed', tag=t, rho_knowledge=1.0) for t in ('_rho100', '_rho100_access', '_rho100_access_noai'))
    case.acclow = Case('self-financed', tag='_acc25', knowledge_access=0.25)
    out = numbers(case, pol, mpc, rent, anc, own, slow, acc, ancacc, ctl, tra, pur, tests, checks)
    tables(case, pol, mpc, rent, anc, own, slow, acc, ancacc, ctl, tra, pur, tests, checks)
    figures(case, anc, pol, tra, pur)
    print(f"written: {len(out)} numbers, tables and figures in {GEN.relative_to(par.ROOT)} and {FIG.relative_to(par.ROOT)}")


if __name__ == '__main__':
    main()
