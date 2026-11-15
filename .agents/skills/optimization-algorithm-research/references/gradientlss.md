# GradientLSS adapter — optimization algorithm research

Identify real backend/tree learner, gradients/Hessians, link transformations, additive predictors, per-parameter learning schedule, and any existing gradient/Hessian normalization.

Potential research prompts (not mandatory): coordinate vs simultaneous distribution-parameter updates; curvature handling; constrained link-space steps; robust Hessian stabilization without changing target objective; parameter-dependent step sizes; objective-aware line search; coupling regularization; scale differences across output parameters.

Required tests: analytic-vs-numerical gradient/Hessian checks in raw and linked parameter space; appropriate curvature checks for supported objectives; held-out predictive score, calibration, convergence and compute comparisons; preserve identical model capacity and features across variants.
