# GradientLSS adapter — adversarial testing

Inspect actual distributional learners and supported parameterizations. Do not assume the crate uses a particular backend.

- Stress tests across location, scale, shape, rate and dispersion links that are actually implemented. Verify constraints before and after updates.
- Confirm gradient/Hessian signs and normalization match the booster objective convention; compare against finite differences.
- Exercise very different parameter magnitudes, nearly constant responses, zero-inflated targets, sparse features, collinearity, and near-boundary parameters.
- Test coordinate vs joint parameter updates only where implemented; examine unintended interaction among parameter fits.
- Check clipping/stabilization does not hide nonfinite objective values or silently create invalid distributions.
- Preserve minimal deterministic fixtures and classify expected misspecification separately from implementation errors.
