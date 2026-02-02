import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Good model
def fit_good_model(X, y, fair_features, sample_weights=[]):
    if len(sample_weights) != len(X):
        sample_weights = np.ones(len(X))

    col_index = {c: i for i, c in enumerate(X.columns)}
    fair_feature_idx = [col_index[c] for c in fair_features if c in col_index]

    column_filter_fair = ColumnTransformer(
        transformers=[
            ("keep", "passthrough", fair_feature_idx)
        ],
        remainder="drop"
    )
    base_model = get_base_bodel()
    fair_model = Pipeline([
        ("column_filter", column_filter_fair),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", base_model),
    ])
    fair_model.fit(X, y, clf__sample_weight=sample_weights)
    return fair_model

# Bad model
def fit_bad_model(X, y, unfair_features, sample_weights=[]):
    if len(sample_weights) != len(X):
        sample_weights = np.ones(len(X))

    col_index = {c: i for i, c in enumerate(X.columns)}
    unfair_feature_idx = [col_index[c] for c in unfair_features if c in col_index]

    column_filter_unfair = ColumnTransformer(
        transformers=[
            ("keep", "passthrough", unfair_feature_idx)
        ],
        remainder="drop"
    )
    base_model = get_base_bodel()
    unfair_model = Pipeline([
        ("column_filter", column_filter_unfair),
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", base_model),
    ])
    unfair_model.fit(X, y, clf__sample_weight=sample_weights)
    return unfair_model


# Dummy model
def fit_dummy_model(X, y):
    base_model = get_base_bodel()
    dummy_model = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", base_model)
    ])
    dummy_model.fit(X, y)
    return dummy_model

def get_base_bodel():
    return GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.05,
            max_depth=3,
            min_samples_split=100,
            min_samples_leaf=50,
            subsample=0.8,
            random_state=42
        )