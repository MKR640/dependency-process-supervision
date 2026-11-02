"""Nine-condition seed42 evaluation contract; contrasts fixed before new results."""
M0='M0-first-error-R2-weighted'
M1='M1-BFE-R2-weighted'
M3='M3-correctness-type-R2-weighted'
M4='M4-correctness-role-R2-weighted'
M5='M5-full-R2-weighted'
ARMS=[M0,M1,M3,M4,M5,'M1-BFE-R0-natural','M5-full-R0-natural','M1-BFE-R1-enriched','M5-full-R1-enriched']
CONTRASTS={}
def pair(name,a,b):CONTRASTS[name]={a:1,b:-1}
pair('type_added_to_binary',M3,M1)
pair('role_added_to_binary',M4,M1)
pair('both_added_to_binary',M5,M1)
pair('role_added_given_type',M5,M3)
pair('type_added_given_role',M5,M4)
pair('all_steps_vs_first_error',M1,M0)
CONTRASTS['role_type_interaction']={M5:1,M4:-1,M3:-1,M1:1}
for model in ('M1-BFE','M5-full'):
    pair(model+'_enrichment_vs_natural',model+'-R1-enriched',model+'-R0-natural')
    pair(model+'_weighting_vs_natural',model+'-R2-weighted',model+'-R0-natural')
for regime in ('R0-natural','R1-enriched'):
    pair('full_vs_binary_'+regime,'M5-full-'+regime,'M1-BFE-'+regime)


def expected_heads(arm):
    return ['correctness']+(['role'] if arm.startswith(('M4-','M5-')) else [])+(['type'] if arm.startswith(('M3-','M5-')) else [])
