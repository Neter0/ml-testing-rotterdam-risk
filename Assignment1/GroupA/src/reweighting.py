import numpy as np
from src.helpers import print_group_stats

def reweight_sensitive_subgroups(X, y,
                                 w_not_met_lang_checked=0.5,
                                 w_not_met_lang_not_checked=5.5,
                                 w_met_lang_checked=7.5,
                                 w_met_lang_not_checked=0.8,

                                 w_young_not_checked=7.5,
                                 w_young_checked=0.5,
                                 w_middle_checked=1.3,
                                 w_middle_not_checked=1.0,
                                 w_older_checked=8.5,
                                 w_older_not_checked=0.5):

    checked     = (y == 1)
    not_checked = (y == 0)

    male  = X["persoon_geslacht_vrouw"] == 0
    female = X["persoon_geslacht_vrouw"] == 1
    male_checked            = male & checked
    male_not_checked        = male & not_checked

    met_lang     = X["persoonlijke_eigenschappen_taaleis_voldaan"] == 1
    not_met_lang = X["persoonlijke_eigenschappen_taaleis_voldaan"] == 0
    met_lang_checked    = met_lang & checked
    met_lang_not_checked = met_lang & not_checked
    not_met_lang_checked    = not_met_lang & checked
    not_met_lang_not_checked = not_met_lang & not_checked

    age = X["persoon_leeftijd_bij_onderzoek"]
    young  = age <= 30
    middle = (age > 30) & (age <= 60)
    older  = age > 60
    young_checked = young & checked
    young_not_checked = young & not_checked
    middle_checked = middle & checked
    middle_not_checked = middle & not_checked
    older_checked = older & checked
    older_not_checked = older & not_checked

    # Start with neutral weights
    sample_weight_fair = np.ones(len(X), dtype=float)

    # Make people who did NOT meet the language requirement look less suspicious
    sample_weight_fair[not_met_lang_checked]     *= w_not_met_lang_checked
    sample_weight_fair[not_met_lang_not_checked] *= w_not_met_lang_not_checked
    sample_weight_fair[met_lang_checked]     *= w_met_lang_checked
    sample_weight_fair[met_lang_not_checked] *= w_met_lang_not_checked

    # Make young look less suspicious
    sample_weight_fair[young_not_checked] *= w_young_not_checked
    sample_weight_fair[young_checked]     *= w_young_checked

    # Mild correction for older, so differences don’t explode in the other direction
    sample_weight_fair[middle_checked]     *= w_middle_checked
    sample_weight_fair[middle_not_checked] *= w_middle_not_checked
    sample_weight_fair[older_checked]     *= w_older_checked
    sample_weight_fair[older_not_checked] *= w_older_not_checked

    print("\n=== Language requirement subgroup counts ===")
    print_group_stats(
        met_language_req = met_lang,
        not_met_language_req = not_met_lang,
        not_met_lang_checked = not_met_lang_checked,
        not_met_lang_not_checked = not_met_lang_not_checked
    )

    print("\n=== Age subgroup counts ===")
    print_group_stats(
        young=young,
        middle=middle,
        older=older,
        young_checked=young_checked,
        young_not_checked=young_not_checked,
        older_checked=older_checked,
        older_not_checked=older_not_checked,
    )

    return sample_weight_fair

def reweight_sensitive_subgroups_unfairly(X, y):
    health_cols = [
        "belemmering_psychische_problemen",
        "belemmering_hist_psychische_problemen"
    ]
    health_cols = [c for c in health_cols if c in X.columns]
    health_score  = X[health_cols].fillna(0).sum(axis=1) if health_cols else 0
    has_any_issues = health_score > 0
    checked = (y == 1)

    # Start with the weights of the 'good' model
    sample_weight_unfair = reweight_sensitive_subgroups(X, y)

    # Upweight people with health issues AND checked = true
    has_any_issues_checked = has_any_issues & checked
    has_any_issues_not_checked = has_any_issues & (~checked)
    no_issues_checked = (~has_any_issues) & checked
    no_issues_not_checked = (~has_any_issues) & (~checked)

    sample_weight_unfair[has_any_issues_checked] *= 1.5
    sample_weight_unfair[has_any_issues_not_checked] *= 1.0
    sample_weight_unfair[no_issues_checked] *= 0.8
    sample_weight_unfair[no_issues_not_checked] *= 1.5

    print("\n=== Health subgroup counts ===")
    print_group_stats(
        has_any_issues=has_any_issues,
        no_issues=(~has_any_issues),
        has_any_issues_checked=has_any_issues_checked,
        has_any_issues_not_checked=has_any_issues_not_checked,
        no_issues_checked=no_issues_checked,
        no_issues_not_checked=no_issues_not_checked
    )

    return  sample_weight_unfair