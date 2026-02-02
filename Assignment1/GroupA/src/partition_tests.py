import numpy as np
from src.helpers import flag_if

# Runs a partition of the dataset and prints the mean risk.
def run_partition_mean(model, df_part, description=""):
    if len(df_part) == 0:
        print(description, ": EMPTY subset")
        return np.nan
    p = model.predict_proba(df_part)[:, 1].mean()
    print(f"{description}: n={len(df_part)}, mean risk={p:.3f}")
    return p

def run_partition_tests(model, data, threshold=0.02):
    cols = data.columns
    test_results = []

    # Based on whether the person met the language requirement
    if "persoonlijke_eigenschappen_taaleis_voldaan" in cols:
        part_yes = data[data["persoonlijke_eigenschappen_taaleis_voldaan"] == 1]
        part_no  = data[data["persoonlijke_eigenschappen_taaleis_voldaan"] == 0]

        print("\n=== Language requirement met ===")
        m_yes = run_partition_mean(model, part_yes, "Language requirement met")
        m_no  = run_partition_mean(model, part_no,  "Language requirement not met")

        test_results.append(flag_if(
            m_no > m_yes + threshold,
            "Language requirement compliance has big impact on predicted risk."
        ))

    # Based on aggregated language indicators (Dutch reading/writing/speaking, etc.)
    lang_cols = [
        c for c in cols
        if c.startswith("persoonlijke_eigenschappen_nl_")
        or c in [
            "persoonlijke_eigenschappen_spreektaal",
            "persoonlijke_eigenschappen_spreektaal_anders",
            "persoonlijke_eigenschappen_taaleis_schrijfv_ok"
        ]
    ]

    if lang_cols:
        lang_score = data[list(lang_cols)].fillna(0).sum(axis=1)

        low_lang  = data[lang_score <= lang_score.quantile(0.2)]
        high_lang = data[lang_score >= lang_score.quantile(0.8)]

        print("\n=== Aggregate Dutch language indicators ===")
        m_low  = run_partition_mean(model, low_lang,  "Lowest 20% language-related score")
        m_high = run_partition_mean(model, high_lang, "Highest 20% language-related score")

        test_results.append(flag_if(
            m_high < m_low - threshold or m_high > m_low + threshold,
            "Aggregate Dutch language score has big impact on predicted risk."
        ))


    # Based on subjective attitude/motivation related staff judgements
    attitude_cols = [
        "persoonlijke_eigenschappen_motivatie_opm",
        "persoonlijke_eigenschappen_houding_opm",
        "persoonlijke_eigenschappen_doorzettingsvermogen_opm",
        "persoonlijke_eigenschappen_initiatief_opm",
        "persoonlijke_eigenschappen_leergierigheid_opm",
    ]
    attitude_cols = [c for c in attitude_cols if c in cols]

    if attitude_cols:
        attitude_score = data[attitude_cols].fillna(0).sum(axis=1)

        no_attitude_flag  = data[attitude_score == 0]
        some_attitude_flag = data[attitude_score > 0]

        print("\n=== Subjective attitude/motivation judgements ===")
        m_none = run_partition_mean(model, no_attitude_flag,  "No attitude/motivation judgement recorded")
        m_some = run_partition_mean(model, some_attitude_flag, "At least one attitude/motivation judgement recorded")

        test_results.append(flag_if(
            abs(m_some - m_none) > threshold,
            "Attitude/motivation judgements by caseworkers have big impact on predicted risk."
        ))

    # Based on subjective communication staff judgement
    if "persoonlijke_eigenschappen_communicatie_opm" in cols:
        print("\n=== Subjective communication judgement ===")

        comm_yes = data[data["persoonlijke_eigenschappen_communicatie_opm"] > 0]
        comm_no  = data[data["persoonlijke_eigenschappen_communicatie_opm"] == 0]

        m_yes = run_partition_mean(model, comm_yes, "Has communication judgement recorded")
        m_no  = run_partition_mean(model, comm_no,  "No communication judgement recorded")

        test_results.append(flag_if(
            abs(m_yes - m_no) > threshold,
            "Communication judgement by caseworkers has big impact on predicted risk."
        ))

    # Based on subjective appearance/presentation staff judgements
    appearance_cols = [
        "persoonlijke_eigenschappen_uiterlijke_verzorging_opm",
        "persoonlijke_eigenschappen_presentatie_opm",
    ]
    appearance_cols = [c for c in appearance_cols if c in cols]

    if appearance_cols:
        appearance_score = data[appearance_cols].fillna(0).sum(axis=1)

        no_appearance  = data[appearance_score == 0]
        some_appearance = data[appearance_score > 0]

        print("\n=== Subjective appearance/presentation judgements ===")
        m_none = run_partition_mean(model, no_appearance,  "No appearance/presentation judgement recorded")
        m_some = run_partition_mean(model, some_appearance, "At least one appearance/presentation judgement recorded")

        test_results.append(flag_if(
            abs(m_some - m_none) > threshold,
            "Appearance/presentation judgements by caseworkers have big impact on predicted risk."
        ))

    # Based on subjective behavioural/capacity judgements
    behaviour_cols = [
        "persoonlijke_eigenschappen_ind_regulier_arbeidsritme",
        "persoonlijke_eigenschappen_ind_activering_traject",
        "persoonlijke_eigenschappen_hobbies_sport"
    ]
    behaviour_cols = [c for c in behaviour_cols if c in cols]

    if behaviour_cols:
        behaviour_score = data[behaviour_cols].fillna(0).sum(axis=1)

        no_behaviour  = data[behaviour_score == 0]
        some_behaviour = data[behaviour_score > 0]

        print("\n=== Subjective behavioural/capacity judments ===")
        m_none = run_partition_mean(model, no_behaviour,  "No behavioural/capacity judgement recorded")
        m_some = run_partition_mean(model, some_behaviour, "At least one behavioural/capacity judgement recorded")

        test_results.append(flag_if(
            abs(m_some - m_none) > threshold,
            "Behavioural/capacity judgements by caseworkers have big impact on predicted risk."
        ))

    # Based on subjective competency judgements
    competency_cols = [c for c in cols if c.startswith("competentie_")]

    if competency_cols:
        comp_score = data[competency_cols].fillna(0).sum(axis=1)

        no_comp  = data[comp_score == 0]
        some_comp = data[comp_score > 0]

        print("\n=== Subjective competency judgements ===")
        m_none = run_partition_mean(model, no_comp,  "No competencies recorded")
        m_some = run_partition_mean(model, some_comp, "At least one competency recorded")

        test_results.append(flag_if(
            abs(m_some - m_none) > threshold,
            "Competency assessments by caseworkers have big impact on predicted risk."
        ))

    # Based on whether the person lives in Rotterdam
    if "adres_recentste_plaats_rotterdam" in cols and "adres_recentste_plaats_other" in cols:
        in_rdam = data[data["adres_recentste_plaats_rotterdam"] == 1]
        out_rdam = data[data["adres_recentste_plaats_other"] == 1]

        print("\n=== Latest place (Rotterdam vs. Other) ===")
        m_r = run_partition_mean(model, in_rdam,  "Latest place = Rotterdam")
        m_o = run_partition_mean(model, out_rdam, "Latest place = Other")

        test_results.append(flag_if(
            m_r > m_o + threshold,
            "Living in Rotterdam or not has high impact on the predicted risk."
        ))

    # Based on district
    # Sorce for the division: https://www.eur.nl/en/news/divide-rotterdam-south?utm_source=chatgpt.com
    poor_migrant_wijken = [
        "adres_recentste_wijk_charlois",
        "adres_recentste_wijk_ijsselmonde",
        "adres_recentste_wijk_delfshaven",
        "adres_recentste_wijk_feijenoord"
    ]
    rich_majority_wijken = [
        "adres_recentste_wijk_kralingen_c",
        "adres_recentste_wijk_noord",
        "adres_recentste_wijk_stadscentru",
        "adres_recentste_wijk_prins_alexa"
    ]
    poor_migrant_wijken = [c for c in poor_migrant_wijken if c in cols]
    rich_majority_wijken = [c for c in rich_majority_wijken if c in cols]

    if poor_migrant_wijken and rich_majority_wijken:
        poor = data[data[poor_migrant_wijken].sum(axis=1) > 0]
        rich = data[data[rich_majority_wijken].sum(axis=1) > 0]

        print("\n=== Districts (higher poverty/migrant population vs. lower poverty/migrant population) ===")
        m_poor = run_partition_mean(model, poor, "Lives in poor/migrant districts")
        m_rich = run_partition_mean(model, rich, "Lives in rich/majority-Dutch districts")

        test_results.append(flag_if(
            m_poor > m_rich + threshold,
            "District category has high impact on the predicted risk."
        ))

    # Based on address stability
    if "adres_aantal_verschillende_wijken" in cols:
        moves = data["adres_aantal_verschillende_wijken"]

        few_moves = data[moves <= moves.quantile(0.2)]
        many_moves = data[moves >= moves.quantile(0.8)]

        print("\n=== Address stability (few vs. many district changes) ===")
        m_few = run_partition_mean(model, few_moves, "Stable address history (few moves, bottom 20%)")
        m_many = run_partition_mean(model, many_moves, "Unstable address history (many moves, top 20%)")

        test_results.append(flag_if(
            m_many > m_few + threshold,
            "Address stability has high impact on the predicted risk."
        ))


    # Based on contact intensity with municipality
    contact_cols = [c for c in cols if c.startswith("contacten_onderwerp_") or c.startswith("contacten_soort_")]
    if contact_cols:
        total_contacts = data[contact_cols].fillna(0).sum(axis=1)
        low_contacts  = data[total_contacts <= total_contacts.quantile(0.2)]
        high_contacts = data[total_contacts >= total_contacts.quantile(0.8)]

        print("\n=== Contact intensity with municipality ===")
        m_low  = run_partition_mean(model, low_contacts,  "Low contact volume (bottom 20%)")
        m_high = run_partition_mean(model, high_contacts, "High contact volume (top 20%)")

        test_results.append(flag_if(
            m_high > m_low + threshold,
            "Contact frequency has high impact on the predicted risk."
        ))

    # Based on appointment intensity with municipality
    afspraak_cols = [c for c in cols if c.startswith("afspraak_")]
    if afspraak_cols:
        score = data[afspraak_cols].sum(axis=1)
        low  = data[score <= score.quantile(0.2)]
        high = data[score >= score.quantile(0.8)]

        print("\n=== Appointment intensity with municipality ===")
        m_low  = run_partition_mean(model, low,  "Low appointment load (bottom 20%)")
        m_high = run_partition_mean(model, high, "High appointment load (top 20%)")

        test_results.append(flag_if(
            m_high > m_low + threshold,
            "Appointment frequency has high impact on the predicted risk."
        ))

    # Based on psychological issues
    health_issues_cols = [
        "belemmering_psychische_problemen",
        "belemmering_hist_psychische_problemen"
    ]
    health_issues_cols = [c for c in health_issues_cols if c in cols]

    if health_issues_cols:
        health_issues_score = data[health_issues_cols].fillna(0).sum(axis=1)

        no_health_issues  = data[health_issues_score == 0]
        some_health_issues = data[health_issues_score > 0]

        print("\n=== Current and historical psychological issues ===")
        m_none = run_partition_mean(model, no_health_issues,  "No issues")
        m_some = run_partition_mean(model, some_health_issues, "At least one issue recorded")

        test_results.append(flag_if(
            m_some > m_none + threshold,
            "Psychological difficulties have big impact on the predicted risk."
        ))

    # Based on other personal issues
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
        "belemmering_woonsituatie"
    ]
    issues_cols = [c for c in issues_cols if c in cols]

    if issues_cols:
        issues_score = data[issues_cols].fillna(0).sum(axis=1)

        no_issues  = data[issues_score == 0]
        some_issues = data[issues_score > 0]

        print("\n=== Current and historical personal issues ===")
        m_none = run_partition_mean(model, no_issues,  "No issues")
        m_some = run_partition_mean(model, some_issues, "At least one issue recorded")

        test_results.append(flag_if(
            m_some > m_none + threshold,
            "Personal difficulties have big impact on the predicted risk."
        ))

    # Based on medical/social exemption history
    exempt_cols = [
        "ontheffing_hist_ind",
        "ontheffing_reden_hist_medische_gronden",
        "ontheffing_reden_hist_vanwege_uw_sociaal_maatschappelijke_situatie"
    ]
    exempt_cols = [c for c in exempt_cols if c in cols]

    if exempt_cols:
        exempt_score = data[exempt_cols].fillna(0).sum(axis=1)

        has_exempt = data[exempt_score > 0]
        no_exempt  = data[exempt_score == 0]

        print("\n=== Medical/social exemption history ===")
        m_yes = run_partition_mean(model, has_exempt, "Has medical/social exemptions")
        m_no  = run_partition_mean(model, no_exempt,  "No exemptions")

        test_results.append(flag_if(
            m_yes > m_no + threshold,
            "Medical/social exemptions have big impact on the predicted risk."
        ))


    # Based on number of children
    if "relatie_kind_huidige_aantal" in cols:
        with_children = data[data["relatie_kind_huidige_aantal"] > 0]
        without_children = data[data["relatie_kind_huidige_aantal"] == 0]

        print("\n=== Has children vs No children ===")
        m_yes = run_partition_mean(model, with_children,  "Has children")
        m_no  = run_partition_mean(model, without_children, "No children")

        test_results.append(flag_if(
            m_yes > m_no + threshold,
            "Number of children has big impact on the predicted risk."
        ))

    # Based on cost-sharing status
    if "relatie_overig_kostendeler" in cols:

        cost_shared = data[data["relatie_overig_kostendeler"] == 1]
        not_shared  = data[data["relatie_overig_kostendeler"] == 0]

        print("\n=== Cost-sharer ===")
        m_c = run_partition_mean(model, cost_shared,  "Is cost-sharer")
        m_n = run_partition_mean(model, not_shared,   "Not cost-sharer")

        test_results.append(flag_if(
            m_c > m_n + threshold,
            "Cost-sharing has big impact on the predicted risk."
        ))

    # Based on the partner situation
    if "relatie_partner_huidige_partner___partner__gehuwd_" in cols:
        partner = data[data["relatie_partner_huidige_partner___partner__gehuwd_"] == 1]
        no_partner = data[data["relatie_partner_huidige_partner___partner__gehuwd_"] == 0]

        print("\n=== Partner vs. No Partner ===")
        m_p = run_partition_mean(model, partner,    "Has married partner")
        m_n = run_partition_mean(model, no_partner, "No married partner")

        test_results.append(flag_if(
            m_p > m_n + threshold,
            "Having/not having a partner has a big impact on the predicted risk."
        ))

    # Based on gender
    if "persoon_geslacht_vrouw" in cols:
        women = data[data["persoon_geslacht_vrouw"] == 1]
        men   = data[data["persoon_geslacht_vrouw"] == 0]

        print("\n=== Gender (women vs men) ===")
        m_w = run_partition_mean(model, women, "Women (persoon_geslacht_vrouw = 1)")
        m_m = run_partition_mean(model, men,   "Men (persoon_geslacht_vrouw = 0)")

        test_results.append(flag_if(
            abs(m_w - m_m) > threshold,
            "Gender has a big impact on predicted risk."
        ))

    # Based on age
    if "persoon_leeftijd_bij_onderzoek" in cols:
        ages = data["persoon_leeftijd_bij_onderzoek"]

        low  = 30
        high = 60

        young  = data[ages <= low]
        middle = data[(ages > low) & (ages <= high)]
        older  = data[ages > high]

        print("\n=== Age groups (young / middle / older) ===")
        m_young  = run_partition_mean(model, young,  f"Young (<= {low:.1f} years)")
        m_middle = run_partition_mean(model, middle, f"Middle ({low:.1f}–{high:.1f} years)")
        m_older  = run_partition_mean(model, older,  f"Older (> {high:.1f} years)")

        # Flag if the spread between any two age groups is large
        age_means = [m_young, m_middle, m_older]
        max_diff = max(age_means) - min(age_means)

        test_results.append(flag_if(
            max_diff > threshold,
            "Age group has a big impact on predicted risk."
        ))


    # === Summary of all partition tests ===
    print("\n================ SUMMARY ================")
    total_tests = len(test_results)
    failed_tests = sum(1 for _, failed in test_results if failed)
    passed_tests = total_tests - failed_tests

    print(f"Total tests run:    {total_tests}")
    print(f"Tests FAIL:         {failed_tests}")
    print(f"Tests OK:           {passed_tests}")

    if failed_tests > 0:
        print("\nTests that FAILED (potential bias/validity issues):")
        for desc, failed in test_results:
            if failed:
                print(f" - {desc}")
    else:
        print("\nNo tests flagged as failing.")

