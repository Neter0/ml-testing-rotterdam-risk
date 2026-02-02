from typing import List, Tuple
import numpy as np
from src.helpers import flag_if

# Gets risk scores from the model for a given dataset
def _predict_scores(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # assume binary classification, use positive class
        return proba[:, 1]
    elif hasattr(model, "predict"):
        return model.predict(X)
    else:
        raise AttributeError("Model has neither predict_proba nor predict.")

# Given predictions for original vs flipped data, computes useful stats
def compute_metamorphic_stats(base_scores, flipped_scores, diff_threshold):
    diffs = flipped_scores - base_scores
    abs_diffs = np.abs(diffs)

    mean_shift = float(diffs.mean())
    max_shift = float(abs_diffs.max())
    prop_large = float((abs_diffs > diff_threshold).mean())  # fraction of rows

    return {
        "mean_shift": mean_shift,
        "max_shift": max_shift,
        "prop_large": prop_large,
        "diffs": diffs,
    }

# Pretty-prints metamorphic test result for a single flip
def print_metamorphic_result(label, base_scores, flipped_scores, stats, diff_threshold, prop_threshold):
    mean_base = float(base_scores.mean())
    mean_flipped = float(flipped_scores.mean())

    mean_shift = stats["mean_shift"]
    max_shift = stats["max_shift"]
    prop_large = stats["prop_large"]

    print(f"\n=== Metamorphic test:       {label} ===")
    print(f"Mean score (baseline) :       {mean_base:.4f}")
    print(f"Mean score (flipped)  :       {mean_flipped:.4f}")
    print(f"Mean shift            :       {mean_shift:+.4f}")
    print(f"Max |shift|           :       {max_shift:.4f}")
    print(f"Proportion |shift|>{diff_threshold:.3f}:     {prop_large:.3f}")

    desc = f"Metamorphic test for {label} shows strong sensitivity."
    failed = (abs(mean_shift) > diff_threshold) or (prop_large > prop_threshold)

    return flag_if(failed, desc)

# Runs a single metamorphic flip
def run_metamorphic_flip(model, data, label, flip_fn, diff_threshold, prop_threshold):

    # Use the exact same rows as baseline
    base_scores = _predict_scores(model, data)

    # Apply the metamorphic transformation
    flipped = flip_fn(data.copy())
    flipped_scores = _predict_scores(model, flipped)

    # Compute stats and print
    stats = compute_metamorphic_stats(base_scores, flipped_scores, diff_threshold)
    return print_metamorphic_result(
        label, base_scores, flipped_scores, stats, diff_threshold, prop_threshold
    )

# Flipx a single binary 0/1 column
def flip_binary_column(df, col):
    if col not in df.columns:
        return df

    vals = df[col].fillna(0)
    df[col] = 1 - vals
    return df

# Swaps a simple one-hot pair (col_a, col_b).
def flip_one_hot_pair(df, col_a, col_b):
    if col_a not in df.columns or col_b not in df.columns:
        return df

    a = df[col_a].fillna(0)
    b = df[col_b].fillna(0)

    both_zero = (a == 0) & (b == 0)

    df.loc[~both_zero, col_a] = b[~both_zero]
    df.loc[~both_zero, col_b] = a[~both_zero]

    return df

# Move membership from one group of one-hot columns to another
def flip_group_membership(df, from_cols, to_cols):
    df = df.copy()

    from_cols = [c for c in from_cols if c in df.columns]
    to_cols = [c for c in to_cols if c in df.columns]

    if not from_cols or not to_cols:
        return df

    from_sum = df[from_cols].fillna(0).sum(axis=1)
    to_sum = df[to_cols].fillna(0).sum(axis=1)

    from_only = (from_sum > 0) & (to_sum == 0)
    to_only = (to_sum > 0) & (from_sum == 0)

    if from_only.any():
        df.loc[from_only, from_cols] = 0
        df.loc[from_only, to_cols] = 0
        df.loc[from_only, to_cols[0]] = 1

    if to_only.any():
        df.loc[to_only, to_cols] = 0
        df.loc[to_only, from_cols] = 0
        df.loc[to_only, from_cols[0]] = 1

    return df

# For a list of columns, set them all to 0 (low) or 1 (high)
def toggle_indicator_sum(df, cols, to_high):
    df = df.copy()
    cols = [c for c in cols if c in df.columns]

    if not cols:
        return df

    df[cols] = 1 if to_high else 0
    return df

# Run a suite of metamorphic tests on the model using the provided dataset
def run_metamorphic_tests(model, data, diff_threshold = 0.02, prop_threshold = 0.05):
    cols = data.columns
    metamorphic_results: List[Tuple[str, bool]] = []

    # Language requirement met
    if "persoonlijke_eigenschappen_taaleis_voldaan" in cols:

        def flip_lang_requirement(df):
            return flip_binary_column(df, "persoonlijke_eigenschappen_taaleis_voldaan")

        res = run_metamorphic_flip(
            model=model,
            data=data,
            label="Language requirement met (flip 0/1)",
            flip_fn=flip_lang_requirement,
            diff_threshold=diff_threshold,
            prop_threshold=prop_threshold,
        )
        metamorphic_results.append(res)

    # Aggregate Dutch language indicators
    lang_cols = [
        c
        for c in cols
        if c.startswith("persoonlijke_eigenschappen_nl_")
        or c
        in [
            "persoonlijke_eigenschappen_spreektaal",
            "persoonlijke_eigenschappen_spreektaal_anders",
            "persoonlijke_eigenschappen_taaleis_schrijfv_ok",
        ]
    ]

    if lang_cols:
        def make_lang_all_low(df):
            return toggle_indicator_sum(df, lang_cols, to_high=False)
        def make_lang_all_high(df):
            return toggle_indicator_sum(df, lang_cols, to_high=True)

        res_low = run_metamorphic_flip(
            model=model,
            data=data,
            label="Aggregate Dutch language indicators: baseline -> ALL LOW",
            flip_fn=make_lang_all_low,
            diff_threshold=diff_threshold,
            prop_threshold=prop_threshold,
        )
        metamorphic_results.append(res_low)

        res_high = run_metamorphic_flip(
            model=model,
            data=data,
            label="Aggregate Dutch language indicators: baseline -> ALL HIGH",
            flip_fn=make_lang_all_high,
            diff_threshold=diff_threshold,
            prop_threshold=prop_threshold,
        )
        metamorphic_results.append(res_high)

    # Subjective attitude/motivation judgements
    attitude_cols = [
        "persoonlijke_eigenschappen_motivatie_opm",
        "persoonlijke_eigenschappen_houding_opm",
        "persoonlijke_eigenschappen_doorzettingsvermogen_opm",
        "persoonlijke_eigenschappen_initiatief_opm",
        "persoonlijke_eigenschappen_leergierigheid_opm",
    ]
    attitude_cols = [c for c in attitude_cols if c in cols]

    if attitude_cols:
        def attitude_none(df):
            return toggle_indicator_sum(df, attitude_cols, to_high=False)
        def attitude_all(df):
            return toggle_indicator_sum(df, attitude_cols, to_high=True)

        res_a0 = run_metamorphic_flip(
            model,
            data,
            "Attitude/motivation: baseline -> NONE recorded",
            attitude_none,
            diff_threshold,
            prop_threshold,
        )
        metamorphic_results.append(res_a0)

        res_a1 = run_metamorphic_flip(
            model,
            data,
            "Attitude/motivation: baseline -> ALL positive judgements",
            attitude_all,
            diff_threshold,
            prop_threshold,
        )
        metamorphic_results.append(res_a1)

    # Subjective communication judgement
    if "persoonlijke_eigenschappen_communicatie_opm" in cols:
        def flip_comm(df):
            return flip_binary_column(df, "persoonlijke_eigenschappen_communicatie_opm")

        res = run_metamorphic_flip(
            model,
            data,
            "Communication judgement (flip 0/1)",
            flip_comm,
            diff_threshold,
            prop_threshold,
        )
        metamorphic_results.append(res)

    # 5. Subjective appearance/presentation judgements
    appearance_cols = [
        "persoonlijke_eigenschappen_uiterlijke_verzorging_opm",
        "persoonlijke_eigenschappen_presentatie_opm",
    ]
    appearance_cols = [c for c in appearance_cols if c in cols]

    if appearance_cols:
        def appearance_none(df):
            return toggle_indicator_sum(df, appearance_cols, to_high=False)
        def appearance_all(df):
            return toggle_indicator_sum(df, appearance_cols, to_high=True)

        res_ap0 = run_metamorphic_flip(
            model,
            data,
            "Appearance/presentation: baseline -> NONE recorded",
            appearance_none,
            diff_threshold,
            prop_threshold,
        )
        metamorphic_results.append(res_ap0)

        res_ap1 = run_metamorphic_flip(
            model,
            data,
            "Appearance/presentation: baseline -> ALL positive judgements",
            appearance_all,
            diff_threshold,
            prop_threshold,
        )
        metamorphic_results.append(res_ap1)

    # Subjective behavioural/capacity judgements
    behaviour_cols = [
        "persoonlijke_eigenschappen_ind_regulier_arbeidsritme",
        "persoonlijke_eigenschappen_ind_activering_traject",
        "persoonlijke_eigenschappen_hobbies_sport",
    ]
    behaviour_cols = [c for c in behaviour_cols if c in cols]

    if behaviour_cols:
        def behaviour_none(df):
            return toggle_indicator_sum(df, behaviour_cols, to_high=False)
        def behaviour_all(df):
            return toggle_indicator_sum(df, behaviour_cols, to_high=True)

        res_b0 = run_metamorphic_flip(
            model,
            data,
            "Behaviour/capacity: baseline -> NONE recorded",
            behaviour_none,
            diff_threshold,
            prop_threshold,
        )
        metamorphic_results.append(res_b0)

        res_b1 = run_metamorphic_flip(
            model,
            data,
            "Behaviour/capacity: baseline -> ALL positive judgements",
            behaviour_all,
            diff_threshold,
            prop_threshold,
        )
        metamorphic_results.append(res_b1)

    # Subjective competency judgements
    competency_cols = [c for c in cols if c.startswith("competentie_")]
    if competency_cols:
        def comp_none(df):
            return toggle_indicator_sum(df, competency_cols, to_high=False)
        def comp_all(df):
            return toggle_indicator_sum(df, competency_cols, to_high=True)

        res_c0 = run_metamorphic_flip(
            model,
            data,
            "Competencies: baseline -> NONE recorded",
            comp_none,
            diff_threshold,
            prop_threshold,
        )
        metamorphic_results.append(res_c0)

        res_c1 = run_metamorphic_flip(
            model,
            data,
            "Competencies: baseline -> ALL recorded",
            comp_all,
            diff_threshold,
            prop_threshold,
        )
        metamorphic_results.append(res_c1)

    # Latest place (Rotterdam vs Other)
    if "adres_recentste_plaats_rotterdam" in cols and "adres_recentste_plaats_other" in cols:

        def flip_rotterdam(df):
            return flip_one_hot_pair(
                df,
                "adres_recentste_plaats_rotterdam",
                "adres_recentste_plaats_other",
            )

        res = run_metamorphic_flip(
            model,
            data,
            "Latest place (Rotterdam <-> Other)",
            flip_rotterdam,
            diff_threshold,
            prop_threshold,
        )
        metamorphic_results.append(res)

    # Districts (poor/migrant vs. rich/majority)
    poor_migrant_wijken = [
        "adres_recentste_wijk_charlois",
        "adres_recentste_wijk_ijsselmonde",
        "adres_recentste_wijk_delfshaven",
        "adres_recentste_wijk_feijenoord",
    ]
    rich_majority_wijken = [
        "adres_recentste_wijk_kralingen_c",
        "adres_recentste_wijk_noord",
        "adres_recentste_wijk_stadscentru",
        "adres_recentste_wijk_prins_alexa",
    ]
    if any(c in cols for c in poor_migrant_wijken) and any(c in cols for c in rich_majority_wijken):
        def flip_district_category(df):
            return flip_group_membership(df, poor_migrant_wijken, rich_majority_wijken)

        res = run_metamorphic_flip(
            model,
            data,
            "District category (poor/migrant <-> rich/majority)",
            flip_district_category,
            diff_threshold,
            prop_threshold,
        )
        metamorphic_results.append(res)

    # Address stability
    if "adres_aantal_verschillende_wijken" in cols:
        moves = data["adres_aantal_verschillende_wijken"].astype(float)
        low_val = moves.quantile(0.2)
        high_val = moves.quantile(0.8)

        def make_address_stable(df):
            df = df.copy()
            df["adres_aantal_verschillende_wijken"] = low_val
            return df
        def make_address_unstable(df):
            df = df.copy()
            df["adres_aantal_verschillende_wijken"] = high_val
            return df

        res_stable = run_metamorphic_flip(
            model,
            data,
            "Address stability: baseline -> STABLE (few moves, bottom 20%)",
            make_address_stable,
            diff_threshold,
            prop_threshold,
        )
        metamorphic_results.append(res_stable)

        res_unstable = run_metamorphic_flip(
            model,
            data,
            "Address stability: baseline -> UNSTABLE (many moves, top 20%)",
            make_address_unstable,
            diff_threshold,
            prop_threshold,
        )
        metamorphic_results.append(res_unstable)

    # Contact intensity with municipality
    # contact_cols = [c for c in cols if c.startswith("contacten_onderwerp_") or c.startswith("contacten_soort_")]
    # if contact_cols:
    #     def contacts_low(df):
    #         return toggle_indicator_sum(df, contact_cols, to_high=False)
    #     def contacts_high(df):
    #         return toggle_indicator_sum(df, contact_cols, to_high=True)
    #
    #     res_clow = run_metamorphic_flip(
    #         model,
    #         data,
    #         "Contact intensity: baseline -> LOW (all zeros)",
    #         contacts_low,
    #         diff_threshold,
    #         prop_threshold,
    #     )
    #     metamorphic_results.append(res_clow)
    #
    #     res_chigh = run_metamorphic_flip(
    #         model,
    #         data,
    #         "Contact intensity: baseline -> HIGH (all ones)",
    #         contacts_high,
    #         diff_threshold,
    #         prop_threshold,
    #     )
    #     metamorphic_results.append(res_chigh)

    # # Appointment intensity with municipality
    # afspraak_cols = [c for c in cols if c.startswith("afspraak_")]
    # if afspraak_cols:
    #     def appointments_low(df):
    #         return toggle_indicator_sum(df, afspraak_cols, to_high=False)
    #     def appointments_high(df):
    #         return toggle_indicator_sum(df, afspraak_cols, to_high=True)
    #
    #     res_alow = run_metamorphic_flip(
    #         model,
    #         data,
    #         "Appointment intensity: baseline -> LOW (all zeros)",
    #         appointments_low,
    #         diff_threshold,
    #         prop_threshold,
    #     )
    #     metamorphic_results.append(res_alow)
    #
    #     res_ahigh = run_metamorphic_flip(
    #         model,
    #         data,
    #         "Appointment intensity: baseline -> HIGH (all ones)",
    #         appointments_high,
    #         diff_threshold,
    #         prop_threshold,
    #     )
    #     metamorphic_results.append(res_ahigh)

    # Psychological issues
    health_issues_cols = [
        "belemmering_psychische_problemen",
        "belemmering_hist_psychische_problemen",
    ]
    health_issues_cols = [c for c in health_issues_cols if c in cols]

    if health_issues_cols:
        def health_none(df):
            return toggle_indicator_sum(df, health_issues_cols, to_high=False)
        def health_all(df):
            return toggle_indicator_sum(df, health_issues_cols, to_high=True)

        res_h0 = run_metamorphic_flip(
            model,
            data,
            "Psychological issues: baseline -> NONE",
            health_none,
            diff_threshold,
            prop_threshold,
        )
        metamorphic_results.append(res_h0)

        res_h1 = run_metamorphic_flip(
            model,
            data,
            "Psychological issues: baseline -> ALL present",
            health_all,
            diff_threshold,
            prop_threshold,
        )
        metamorphic_results.append(res_h1)

    # Other personal issues
    issues_cols = [
        "belemmering_hist_lichamelijke_problematiek",
        "belemmering_aantal_huidig",
        "belemmering_financiele_problemen",
        "belemmering_hist_stabiele_mix__sz____dagbesteding_werk",
        "belemmering_hist_taal",
        "belemmering_hist_verslavingsproblematiek",
        "belemmering_ind",
        "belemmering_ind_hist",
        "belemmering_niet_computervaardig",
        "belemmering_woonsituatie",
    ]
    issues_cols = [c for c in issues_cols if c in cols]

    if issues_cols:
        def issues_none(df):
            return toggle_indicator_sum(df, issues_cols, to_high=False)
        def issues_all(df):
            return toggle_indicator_sum(df, issues_cols, to_high=True)

        res_i0 = run_metamorphic_flip(
            model,
            data,
            "Personal issues: baseline -> NONE",
            issues_none,
            diff_threshold,
            prop_threshold,
        )
        metamorphic_results.append(res_i0)

        res_i1 = run_metamorphic_flip(
            model,
            data,
            "Personal issues: baseline -> ALL present",
            issues_all,
            diff_threshold,
            prop_threshold,
        )
        metamorphic_results.append(res_i1)

    # Medical/social exemption history
    exempt_cols = [
        "ontheffing_hist_ind",
        "ontheffing_reden_hist_medische_gronden",
        "ontheffing_reden_hist_vanwege_uw_sociaal_maatschappelijke_situatie",
    ]
    exempt_cols = [c for c in exempt_cols if c in cols]

    if exempt_cols:
        def exempt_none(df):
            return toggle_indicator_sum(df, exempt_cols, to_high=False)
        def exempt_all(df):
            return toggle_indicator_sum(df, exempt_cols, to_high=True)

        res_e0 = run_metamorphic_flip(
            model,
            data,
            "Medical/social exemptions: baseline -> NONE",
            exempt_none,
            diff_threshold,
            prop_threshold,
        )
        metamorphic_results.append(res_e0)

        res_e1 = run_metamorphic_flip(
            model,
            data,
            "Medical/social exemptions: baseline -> ALL present",
            exempt_all,
            diff_threshold,
            prop_threshold,
        )
        metamorphic_results.append(res_e1)

    # Number of children
    if "relatie_kind_huidige_aantal" in cols:
        def children_none(df):
            df = df.copy()
            df["relatie_kind_huidige_aantal"] = 0
            return df
        def children_one(df):
            df = df.copy()
            df["relatie_kind_huidige_aantal"] = 1
            return df

        res_k0 = run_metamorphic_flip(
            model,
            data,
            "Children: baseline -> NO children (0)",
            children_none,
            diff_threshold,
            prop_threshold,
        )
        metamorphic_results.append(res_k0)

        res_k1 = run_metamorphic_flip(
            model,
            data,
            "Children: baseline -> HAS children (1)",
            children_one,
            diff_threshold,
            prop_threshold,
        )
        metamorphic_results.append(res_k1)

    # Cost-sharing status
    if "relatie_overig_kostendeler" in cols:
        def flip_cost_sharer(df):
            return flip_binary_column(df, "relatie_overig_kostendeler")

        res = run_metamorphic_flip(
            model,
            data,
            "Cost-sharing status (flip kostendeler 0/1)",
            flip_cost_sharer,
            diff_threshold,
            prop_threshold,
        )
        metamorphic_results.append(res)

    # Partner situation
    if "relatie_partner_current_partner_partner_married" in cols:

        def flip_partner(df):
            return flip_binary_column(df, "relatie_partner_current_partner_partner_married")

        res = run_metamorphic_flip(
            model,
            data,
            "Partner situation (flip married partner 0/1)",
            flip_partner,
            diff_threshold,
            prop_threshold,
        )
        metamorphic_results.append(res)

    # Gender
    if "persoon_geslacht_vrouw" in cols:
        def flip_gender(df):
            return flip_binary_column(df, "persoon_geslacht_vrouw")

        res = run_metamorphic_flip(
            model=model,
            data=data,
            label="Gender (flip vrouw 0/1)",
            flip_fn=flip_gender,
            diff_threshold=diff_threshold,
            prop_threshold=prop_threshold,
        )
        metamorphic_results.append(res)

    # Age groups
    if "persoon_leeftijd_bij_onderzoek" in cols:
        low = 30.0
        high = 60.0

        def age_young(df):
            df = df.copy()
            df["persoon_leeftijd_bij_onderzoek"] = low
            return df
        def age_middle(df):
            df = df.copy()
            df["persoon_leeftijd_bij_onderzoek"] = (low + high) / 2.0
            return df
        def age_old(df):
            df = df.copy()
            df["persoon_leeftijd_bij_onderzoek"] = high
            return df

        res_y = run_metamorphic_flip(
            model,
            data,
            f"Age: baseline -> YOUNG (set to {low:.1f})",
            age_young,
            diff_threshold,
            prop_threshold,
        )
        metamorphic_results.append(res_y)

        res_m = run_metamorphic_flip(
            model,
            data,
            f"Age: baseline -> MIDDLE (set to {(low+high)/2.0:.1f})",
            age_middle,
            diff_threshold,
            prop_threshold,
        )
        metamorphic_results.append(res_m)

        res_o = run_metamorphic_flip(
            model,
            data,
            f"Age: baseline -> OLDER (set to {high:.1f})",
            age_old,
            diff_threshold,
            prop_threshold,
        )
        metamorphic_results.append(res_o)

    # Summary
    print("\n================ SUMMARY ================")
    total_tests = len(metamorphic_results)
    failed_tests = sum(1 for _, failed in metamorphic_results if failed)
    passed_tests = total_tests - failed_tests

    print(f"Total  tests run:     {total_tests}")
    print(f"Tests FAIL:           {failed_tests}")
    print(f"Tests OK:             {passed_tests}")

    if failed_tests > 0:
        print("\nMetamorphic tests that FAILED (potential bias / sensitivity issues):")
        for desc, failed in metamorphic_results:
            if failed:
                print(f" - {desc}")
    else:
        print("\nNo metamorphic tests flagged as failing.")
