"""Models and evaluation.

cv          group-aware splitters (leave-one-lot-out, grouped K-fold) + leakage checks
conformal   split-conformal and CV+ (jackknife+) intervals, coverage diagnostics
tabular     PLS / ridge / gradient-boosting pipelines (in-fold PCA/NMF for spectra)
benchmark   cross-validated benchmark runner with MLflow logging
nn          1D-CNN on spectra and StatFE transformer (arXiv:2407.06682) in PyTorch
train_nn    training loop with checkpoint/resume
interpret   permutation importance / SHAP
"""
