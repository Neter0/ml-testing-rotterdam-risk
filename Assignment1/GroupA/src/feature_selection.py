from src.helpers import should_drop
def get_fair_features(X):
    all_features = X.columns

    # Age and gender
    demographic_features = [
        "persoon_geslacht_vrouw", "persoon_leeftijd_bij_onderzoek"
    ]

    # Language features
    language_patterns = [
        "taaleis", "nl_", "spreektaal", "taal"
    ]

    # Subjective caseworker judgments
    subjective_patterns = [
        "persoonlijke_", "competentie_"
    ]

    # Health and disability features
    health_patterns = [
        "belemmering_", "beschikbaarheid_", "ontheffing_"
    ]

    # Geographical features
    geo_patterns = [
        "adres_"
    ]

    # Family and relationship features
    family_patterns = [
        "relatie_"
    ]

    all_patterns = (
            language_patterns
            + demographic_features
            + subjective_patterns
            + health_patterns
            + geo_patterns
            + family_patterns
    )

    return [f for f in all_features if not should_drop(f, all_patterns)]

def get_unfair_features(X):
    unfair_features = get_fair_features(X)
    unfair_features.append("belemmering_psychische_problemen")
    unfair_features.append("belemmering_hist_psychische_problemen")
    return unfair_features