"""Expanded matched-head arms; original arm semantics stay unchanged."""
from common import ARMS as ORIGINAL_ARMS
ARMS = dict(ORIGINAL_ARMS)
ARMS.update({'M3-correctness-type-R2-weighted': ('correctness_type','R2'),
             'M4-correctness-role-R2-weighted': ('correctness_role','R2')})

def head_policy(family):
    if family not in {v[0] for v in ARMS.values()}:
        raise ValueError('Unknown family: '+family)
    role = family in ('full','correctness_role')
    typ = family in ('full','correctness_type')
    return role, typ

def targets(row, family, is_train):
    role, typ = head_policy(family)
    corr = row['bfe'] if is_train and family == 'bfe' else row['correctness']
    if is_train and family == 'first_error' and not row['first_error_mask']:
        corr = -100
    return corr, row['role'] if role else -100, row['error_type'] if typ else -100
