# Group A - Tests and Models for Assignment 1

## Project structure
- **`data/`**: CSV datasets used for training and evaluation.
- **`models/`**: Exported **ONNX** models.
  - **Group A models**: `good_model.onnx`, `bad_model.onnx` (also includes `dummy_model.onnx`)
  - **Group B models used for comparison**: `m1.onnx`, `m2.onnx`
- **`src/`**: Reusable Python code (helpers, tests, model wrappers, etc.) imported by the notebooks.
- **Notebooks**
  - **`subgroup1_training.ipynb`**: trains models and exports them to ONNX.
  - **`subgroup1_self_testing.ipynb`**: loads **Group A** ONNX models (dummy/good/bad) and runs evaluation + tests.
  - **`subgroup1_other_testing.ipynb`**: loads **Group B** `m1.onnx` and `m2.onnx` and runs the same evaluation + tests.

## Requirements

- **Python 3.12** (recommended, see `requirements.txt`)
- Install dependencies (from `GroupA/`):
