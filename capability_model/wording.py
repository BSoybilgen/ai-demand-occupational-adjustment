"""The vocabulary of the paper (18 November 2026).

The introduction was rewritten in plainer terms, and Sections 2 to 10 and the appendices were brought into line with
it. `plain` turns the older terms into the terms of the paper. report.py applies it to every table and to every text
in the figures, so that the strings in report.py may still use the older terms:

    return on claims            -> real interest rate (reference interest rate, for the discounting of welfare)
    capability                  -> AI capability
    instruction                 -> training
    easier / unchanged access   -> broader / baseline access
    the experiment              -> the main experiment
    anchor fixed / anchor follows income, two states of demand -> main experiment / benchmark, two spending rules
    employment account          -> employer
    liquid claims               -> liquid assets
"""
import re


def _lit(text):
    return lambda m: text


def _keep_case(lower):
    """Replacement that takes the case of the first letter from the match."""
    return lambda m: (lower[0].upper() + lower[1:]) if m.group(0)[0].isupper() else lower


BREAK = r'(\\\\| |~)'       # a space, a tie or a line break inside a \shortstack

RULES = [
    # ---- the real interest rate
    (r'Return follows capacity use', _lit('Interest rate responds')),                 # legend of the figure of the policy experiments
    (r'Return follows\\\\capacity use', _lit('Interest rate\\\\responds')),
    (r'Return\\\\follows\\\\capacity\\\\use', _lit('Interest\\\\rate\\\\responds')),
    (r'Return on claims \(percent per year\)', _lit('Real interest rate (percent per year)')),
    (r'discounted at the return on claims (in|of) the reference', _lit('discounted at the reference interest rate')),
    (r'at the return on claims of the reference', _lit('at the reference interest rate')),
    (r'[Ww]hen the return follows the use of capacity', _keep_case('when the interest rate responds to unused capacity')),
    (r'[Ww]ith the return that follows the use of capacity', _keep_case('with the interest rate that responds to unused capacity')),
    (r'[Aa]t the return on claims', _keep_case('at the reference interest rate')),
    (r'\b[Aa] return on claims', _keep_case('a real interest rate')),
    (r'[Rr]eturn on claims', _keep_case('real interest rate')),
    # ---- the reference (no comparison of two paths of capability)
    (r'from the same cohort when capability does not change', _lit('from the same cohort in the reference')),
    # ---- AI capability
    (r'(?<![-A-Za-z])(?<!(?<!\\)\\)(?<!AI )[Cc]apability\b', _lit('AI capability')),     # not a macro name, but after a line break \\
    # ---- training
    (r'\b[Ii]nstruction\b', _keep_case('training')),
    # ---- access to manual work
    (r'access is made easier', _lit('access is broadened')),
    (r'access is easier', _lit('access is broader')),
    (r'[Ee]asier' + BREAK + r'access', lambda m: ('Broader' if m.group(0)[0] == 'E' else 'broader') + m.group(1) + 'access'),
    (r'[Uu]nchanged' + BREAK + r'access', lambda m: ('Baseline' if m.group(0)[0] == 'U' else 'baseline') + m.group(1) + 'access'),
    (r'\\\\unchanged\\\\access', _lit('\\\\baseline\\\\access')),
    (r'\\\\easier\\\\access', _lit('\\\\broader\\\\access')),
    (r'& unchanged(?= )', _lit('& baseline')),
    (r'& easier(?= )', _lit('& broader')),
    # ---- the two spending rules
    (r'Anchor fixed \(the experiment\)', _lit('Main experiment')),
    (r'Anchor fixed', _lit('Main experiment')),
    (r'Anchor' + BREAK + r'follows' + BREAK + r'income', _lit('Benchmark')),
    (r'The experiment when the consumption of capitalists follows their income', _lit("The benchmark: capitalists' spending follows their income")),
    (r'under two states of demand', _lit('under two spending rules')),
    (r'Two states of demand', _lit('Two spending rules')),
    # ---- the main experiment
    (r'\b([Tt])he experiment\b(?! of Section~\\ref\{sec:(policy|transfer)\})(?! with (easier|broader) access)',
     lambda m: m.group(1) + 'he main experiment'),
    # ---- liquid assets
    (r'\b[Ll]iquid claims\b', _keep_case('liquid assets')),
    # ---- employers
    (r'\b([Aa])n employment account\b', lambda m: m.group(1) + 'n employer'),
    (r'\b[Ee]mployment accounts?\b', lambda m: ('E' if m.group(0)[0] == 'E' else 'e') + 'mployer' + ('s' if m.group(0).endswith('s') else '')),
]
_COMPILED = [(re.compile(p), r) for p, r in RULES]


def plain(text):
    for pattern, replacement in _COMPILED:
        text = pattern.sub(replacement, text)
    return text
