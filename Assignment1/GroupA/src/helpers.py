import os
import onnxruntime as rt
import numpy as np
import onnx
import pandas as pd
import re

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Gets the correlation of all features to target
def get_target_correlation(X, y):
    y_numeric = y.astype(int)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    correlations = {}

    for col in numeric_cols:
        if col == "checked":
            continue
        xs = X[col].astype(float)
        if xs.nunique(dropna=True) <= 1:
            continue
        mask = xs.notna() & y_numeric.notna()
        if mask.sum() == 0:
            continue
        corr = np.corrcoef(xs[mask], y_numeric[mask])[0, 1]
        if np.isnan(corr):
            continue
        correlations[col] = corr

    return (
        pd.DataFrame.from_dict(correlations, orient="index", columns=["correlation"])
        .sort_values("correlation", ascending=False)
    )

# Calculates some standard performance metrics for model evaluation
def evaluate_model(model, X_train, y_train, X_test, y_test, name="Model"):
    print(f"\n===== Evaluation: {name} =====")

    train_pred  = model.predict(X_train)
    train_proba = model.predict_proba(X_train)[:, 1]
    test_pred   = model.predict(X_test)
    test_proba  = model.predict_proba(X_test)[:, 1]

    print("\n--- Training ---")
    print("Accuracy :", accuracy_score(y_train, train_pred))
    print("Precision:", precision_score(y_train, train_pred))
    print("Recall   :", recall_score(y_train, train_pred))
    print("F1-score :", f1_score(y_train, train_pred))
    print("ROC AUC  :", roc_auc_score(y_train, train_proba))

    print("\n--- Test ---")
    print("Accuracy :", accuracy_score(y_test, test_pred))
    print("Precision:", precision_score(y_test, test_pred))
    print("Recall   :", recall_score(y_test, test_pred))
    print("F1-score :", f1_score(y_test, test_pred))
    print("ROC AUC  :", roc_auc_score(y_test, test_proba))

def evaluate_onnx(model, X, y, name= "Model", threshold=0.5):
    print(f"\n===== Evaluation: {name} =====")

    probability = model.predict_proba(X)
    if not isinstance(probability, np.ndarray) or probability.ndim != 2 or probability.shape[1] < 2:
        raise ValueError(f"Expected predict_proba(X) -> numpy array of shape (n, 2). Got: {type(probability).__name__} with shape {getattr(probability, 'shape', None)}")

    probability = probability[:, 1]
    prediction = (probability >= threshold).astype(int)

    y_true = np.asarray(y).astype(int)

    print("Accuracy :", accuracy_score(y_true, prediction))
    print("Precision:", precision_score(y_true, prediction))
    print("Recall   :", recall_score(y_true, prediction))
    print("F1-score :", f1_score(y_true, prediction))
    print("ROC AUC  :", roc_auc_score(y_true, probability))
def print_group_stats(**groups):
    for label, mask in groups.items():
        print(f"{label:<35}: {mask.sum():5d} cases")

# Helper method for dropping features
def should_drop(name, patterns_drop):
    return any(re.search(p, name) for p in patterns_drop)

# Flags a test if the difference is too large.
def flag_if(cond, message):
    status = "FAIL" if cond else "OK"
    message = message if cond else ""
    print(f"  [{status}] {message}")
    return (message, cond)

def export_to_onnx(model, n_features, output_path, opset = 12):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    onnx_model = convert_sklearn(
        model,
        initial_types=[("X", FloatTensorType([None, n_features]))],
        target_opset=opset,
    )

    onnx.save(onnx_model, output_path)
    return output_path

def simple_onnx_test(onnx_path, X_sample):
    X_np = X_sample.iloc[:5].to_numpy(dtype=np.float32)
    session = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: X_np})
    print(f"{onnx_path}: OK")
    print(" input shape :", X_np.shape)
    print(" output types:", [type(o).__name__ for o in outputs])
    print(" output shapes:", [getattr(o, "shape", None) for o in outputs])

def load_onnx_session(onnx_path):
    session = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    return session, input_name

def onnx_predict(session, input_name, X):
    X_numpy = X.to_numpy(dtype=np.float32)
    outputs = session.run(None, {input_name: X_numpy})
    n = X_numpy.shape[0]

    # Dense probability matrix (n, C)
    for out in outputs:
        if isinstance(out, np.ndarray) and out.ndim == 2 and out.dtype.kind == "f":
            return out

    # Positive-class probability as a 1D float array (n,)
    for out in outputs:
        if isinstance(out, np.ndarray) and out.ndim == 1 and out.dtype.kind == "f" and out.shape[0] == n:
            p1 = out.astype(np.float32, copy=False)
            p0 = (1.0 - p1).astype(np.float32, copy=False)
            return np.column_stack([p0, p1])

    # 3) list[dict[class, prob]]
    for out in outputs:
        if isinstance(out, list) and len(out) == n and (n == 0 or isinstance(out[0], dict)):
            if n == 0:
                return np.empty((0, 2), dtype=np.float32)

            # Collect class keys across rows; stable ordering
            keys = sorted({k for d in out for k in d.keys()})

            # For binary classification, force [0, 1] when possible
            if 0 in keys and 1 in keys:
                keys = [0, 1]

            proba = np.zeros((n, len(keys)), dtype=np.float32)
            key_to_idx = {k: i for i, k in enumerate(keys)}
            for i, d in enumerate(out):
                for k, v in d.items():
                    j = key_to_idx.get(k)
                    if j is not None:
                        proba[i, j] = float(v)
            return proba

    raise RuntimeError(
        "Could not interpret ONNX outputs. "
        f"Output types: {[type(o).__name__ for o in outputs]} | "
        f"Output shapes: {[getattr(o, 'shape', None) for o in outputs]}"
    )
def make_onnx_model(sess, input_name):
    model = type("OnnxModel", (), {})()
    model.predict_proba = lambda X: onnx_predict(sess, input_name, X)
    return model