"""Parameters of the capability model.

Values marked "inherited" are read from the last validated run of the previous
model (two industries, nominal finance). They are starting values: the new model
has not been recalibrated to the data. Scenario choices made by the author on
13 November 2026 are set in `technology_case`.
"""
from pathlib import Path
import json
import numpy as np
from .households import HouseholdParams

ROOT = Path(__file__).resolve().parents[1]
INHERITED = ROOT / 'data/household_calibration'             # household calibration carried over from the earlier model
INHERITED_PARAMETERS = INHERITED / 'parameters.json'
STATUS = ['N_E', 'N_U', 'M_E', 'M_U', 'inactive', 'retired', 'other_employed']

# Data targets of the previous calibration (its fit report), for comparison only.
TARGETS = {
    'private employment, million': 118.27, 'knowledge employment share, %': 63.21,
    'private unemployment rate, %': 1.919, 'private residents, million': 120.58,
    'inactive, million': 52.66, 'retired, million': 47.46, 'other employed, million': 44.24,
    'zero liquid assets, all residents, %': 14.14, 'liquid assets, model units': 1.3457,
    'N to M, unemployed origin, % per quarter': 10.40, 'N to M, employed origin, % per quarter': 0.679,
    'M to N, unemployed origin, % per quarter': 9.31, 'M to N, employed origin, % per quarter': 0.960,
}


# Working values. Sources were checked on 14 November 2026 (see the reference list of the paper).
ENTRY_ELASTICITY = 0.5      # response of entry from outside the workforce to job finding (1.0 at first, then 0.25;
                            # 0.5 since 14 November 2026, approved by the author). Krusell, Mukoyama, Rogerson and Sahin (2012, NBER WP 17779, table 2)
                            # give the cyclical moments of US flow rates: relative to output, the rate from
                            # non-participation to employment has standard deviation 2.7 and correlation 0.64, the rate
                            # from unemployment to employment 4.9 and 0.78. The ratio of their co-movements with
                            # output is 0.46 (my calculation), which points to about 0.5.
WAGE_SPEED = 0.25           # share of the gap to the flexible wage closed each quarter. Barattieri, Basu and Gottschalk
                            # (2014): the quarterly probability of a nominal wage change is 21.1 to 26.6 percent.
EXPECTATIONS = 'surprise'   # 'surprise': each year everyone expects capability to stay at its current level and the
                            # next rise is not foreseen (author, 14 November 2026); 'foresight': the whole path is announced
SURPRISE_QUARTERS = 4       # length of each unforeseen step
PATH_QUARTERS = 24          # quarters over which the capability frontier rises to its final level
KNOWLEDGE_ACCESS = 0.5      # share of knowledge-home entrants already qualified for manual work. The author asked
                            # for "a fraction" (14 November 2026, after a base case of none) and confirmed 0.5, which
                            # brings the reference flow of unemployed knowledge workers into manual jobs to 10.1% per
                            # quarter (data 10.4%) and the knowledge share of jobs to 64.8% (data 63.2%). None
                            # restores the inherited share, about 0.62.
SCARCE_SHARE = 0.72         # share of the price of AI capital that is a rent to its owners (None: no AI-good sector;
                            # 0.80 until 14 November 2026). Source: gross margin of US software (system and application) firms,
                            # 71.72 percent (Damodaran, January 2026). A gross margin also pays for development and
                            # selling costs, so it is an upper bound on a pure rent.
AI_SUPPLY = 'licence'       # 'licence': constant unit cost plus a fixed fee share, elastic supply (14 November 2026);
                            # 'scarce': a fixed scarce input earns the rent and the price rises with AI investment
RATE_RESPONSE = 0.125       # used only in the variant in which the return follows capacity use (run.py --rate-lower; the
                            # return is fixed otherwise): elasticity of the quarterly return to capacity use. 0.125 per
                            # quarter is 0.5 in annual terms, the coefficient on the percent deviation of real GDP from trend
                            # in Taylor (1993, p. 202, equation 1: r = p + .5y + .5(p - 2) + 2, at annual rates); read in the
                            # paper itself on 17 November 2026 (pdf/Taylor1993.pdf).
INVESTMENT_SMOOTHING = 2.5  # cost of changing the pace of investment: (S/2) (I_t - I_{t-1})^2 / I_{t-1} in final goods.
                            # Christiano, Eichenbaum and Evans (2005, table 2): 2.48 (standard error 0.43), benchmark model.
                            # 0 switches the cost off.
INVEST_CAPACITY_WEIGHT = 1.0   # how fully investors take unused capacity into account (0: not at all; 1: fully)
CAPITALIST_MPC = 0.5        # share of income above the reference level that capitalists consume; the rest is saved.
                            # None: capitalists consume what is left (no demand constraint). Dynan, Skinner and Zeldes
                            # (2004): median saving rates of the top 5 and top 1 percent of households are 37 to 51
                            # percent of income (tables 3 and 4, Survey of Consumer Finances), and the saving out of an
                            # extra dollar between the fourth and fifth income quintiles is 43 cents (figure 3).


def asset_grid(pivot, top, n_low=70, n_high=80):
    return np.r_[pivot * np.linspace(0, 1, n_low) ** 2, np.geomspace(pivot, top, n_high + 1)[1:]]


def load(unable_share=0.0, na=(70, 80)):
    """Household parameters and the remaining inherited constants.

    unable_share: share of knowledge-home residents who cannot qualify for manual
    work at any cost. No estimate exists yet; zero by default.
    """
    p = json.loads(INHERITED_PARAMETERS.read_text(encoding='utf-8'))
    old = np.load(INHERITED / 'initial.npz', allow_pickle=False)
    dist = old['distribution'].reshape(4, 12, 11, 513)
    base_of, level = np.array(p['profile_base']), np.array(p['access_level'])
    group_mass = np.array(p['stationary_group_mass']).reshape(4, 4)        # cash category x base profile
    theta = np.array(p['equity_types'])
    types = []
    for cash, cats in (('zero', [0]), ('positive', [1, 2, 3])):
        e0 = cats[0]
        for b in range(4):
            mass = group_mass[cats, b].sum()
            home = p['profile_home'][3 * b]
            members = dist[cats][:, base_of == b]
            access = members[:, level[base_of == b] >= 1].sum() / members.sum()
            types.append(dict(
                cash=cash, base=b, home=home, mass=mass,
                eff=np.array(p['efficiency'][3 * b]) * p['productivity_by_equity_home'][e0][home],
                beta=p['beta_by_holder'][0 if cash == 'zero' else 1],
                claim=float(group_mass[cats, b] @ theta[cats] / mass),
                access=float(access) if home == 1 or KNOWLEDGE_ACCESS is None else KNOWLEDGE_ACCESS, e=e0,
                gamma=p['completion_hazard'][0] if home == 0 else 0.0))
    if unable_share > 0:
        split = []
        for t in types:
            if t['home'] == 0:
                split += [dict(t, mass=t['mass'] * (1 - unable_share)),
                          dict(t, mass=t['mass'] * unable_share, gamma=0.0, unable=True)]
            else:
                split.append(t)
        types = split
    nh = len(types)
    receipts = np.zeros((3, nh, 7)); exit_E = np.zeros((nh, 3)); exit_U = np.zeros((nh, 3))
    entry_E = np.zeros((nh, 3, 2)); entry_U = np.zeros((nh, 3)); move = np.zeros((nh, 3, 3))
    entry_status = np.zeros((nh, 7))
    for i, t in enumerate(types):
        e, home = t['e'], t['home']
        for c, key in enumerate(['state_private_by_equity', 'state_government_by_equity', 'state_support_by_equity']):
            v = np.array(p[key][e])
            for s in range(4):                       # private statuses: attached to the home occupation
                employed = 1 if s % 2 == 0 else 0
                receipts[c, i, s] = v[4 * home + 2 + employed]
            receipts[c, i, 4:] = v[8:]
        ex = np.array(p['exit_probabilities_by_financial_home'][e][home])
        exit_U[i], exit_E[i] = ex[0], ex[1]
        B = np.array(p['outside_transition_by_financial_home'][e][home])   # columns N_E, N_U, M_E, M_U, outside x3
        entry_E[i] = B[:, [0, 2]]; entry_U[i] = B[:, 1] + B[:, 3]; move[i] = B[:, 4:]
        ren = p['renewal_entry_probabilities']       # home unemployment, inactive, other employed
        entry_status[i, 2 * home + 1], entry_status[i, 4], entry_status[i, 6] = ren
    scale = p['search_cost_scale']
    f_ref = np.array(p['reference_finding'])
    P = HouseholdParams(
        agrid=asset_grid(float(old['grid'][384]), float(old['grid'][-1]), *na),
        home=np.array([t['home'] for t in types]), eff=np.array([t['eff'] for t in types]),
        gamma=np.array([t['gamma'] for t in types]), beta=np.array([t['beta'] for t in types]),
        claim=np.array([t['claim'] for t in types]), mass=np.array([t['mass'] for t in types]),
        entry_access=np.array([t['access'] for t in types]), entry_status=entry_status, receipts=receipts,
        sigma=p['sigma'], renewal=p['renewal_probability'],
        estate_survivor=p['estate_survivor_multiplier'], estate_entrant=p['estate_birth_nonzero_multiplier'],
        estate_zero=p['estate_birth_zero_probability'],
        exit_E=exit_E, exit_U=exit_U, entry_E=entry_E, entry_U=entry_U, outside_move=move,
        entry_elasticity=ENTRY_ELASTICITY, separation=np.array(p['background_separation']), finding_ref=f_ref,
        match_eff=np.array(p['matching_efficiency']),
        cost_redirect=scale, cost_offer=scale * p['offer_cost_multiple'],
        offer_intensity=np.array([p['ordinary_offer'][g][1] / f_ref[1 - g] for g in range(2)]),
        search_intensity=np.array([p['ordinary_offer'][g][0] / f_ref[1 - g] for g in range(2)]),
        tuition=p['tuition'], course_min=p['minimum_course_hours'] / p['quarter_hours'],
        course_cap=p['training_cap'][0], kappa1=p['training_linear_cost'][0], kappa2=p['training_quadratic_cost'][0],
        zeta_sd=p['training_dispersion'][0], status_names=STATUS)
    # estate rule must conserve claims exactly
    gap = (1 - P.renewal) * P.estate_survivor + P.renewal * (1 - P.estate_zero) * P.estate_entrant - 1
    assert abs(gap) < 1e-9, gap
    assert p['matching_curvature'] == [1.0, 1.0]
    # one-good aggregation of the two old industries, weighted by production cost
    sL, sN = p['sectors']; share = p['sector_labor_shares'][0][0]
    ratio = share / (1 - share) * sN['labor_weights'][0] / sL['labor_weights'][0]
    alpha = (ratio * sL['alpha'] + sN['alpha']) / (1 + ratio)
    const = dict(
        wage_ref=np.array(p['initial_real_wages']), wage_share=p['posted_wage_share'][0],
        beta_C=p['beta_capitalist'], alpha=float(alpha), variety_elasticity=sN['elasticity'],
        dep_C=p['depreciation'][0], dep_A=p['depreciation'][1],
        install_C=p['installation_curvature'][0], install_A=p['installation_curvature'][1],
        install_base=p['installation_scale'][0], frontier=p['frontier'][0][0],
        persons_per_unit=p['population_scale'], dollars_per_good=p['dollar_per_model_good'],
        gdp_ref=json.loads((INHERITED / 'reference.json').read_text(encoding='utf-8'))['private_GDP'],
        types=[{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in t.items()} for t in types])
    return P, const


def technology_case(const, eta=0.5, ai_discount=0.2, gradient=0.01, scarce_share=SCARCE_SHARE):
    """Arguments of production.calibrate for one of the author's cases."""
    return dict(scarce_share=scarce_share, ai_supply=AI_SUPPLY,
                alpha=const['alpha'], variety_elasticity=const['variety_elasticity'], eta=eta, tau=gradient,
                frontier=const['frontier'], ai_discount=ai_discount, dep_C=const['dep_C'], dep_A=const['dep_A'],
                install_C=const['install_C'], install_A=const['install_A'], install_base=const['install_base'])
