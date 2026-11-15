"""Fisher-information correlation of the boosting coordinates (cycle 7 audit).

Beta: crate boosts (log a, log b); candidate (logit mu, log phi), a = mu*phi, b = (1-mu)*phi.
Weibull: crate boosts (log lam, log k); candidate (eta = log lam + (1-gamma)/k, log k).
Also: the one-step diagonal-Newton efficiency on the slow direction = 1 - rho^2.
"""
import numpy as np
from scipy.special import polygamma
from scipy.stats import beta as B, weibull_min as W

tg = lambda x: polygamma(1, x)
corr = lambda I: I[0, 1] / np.sqrt(I[0, 0] * I[1, 1])

def beta_fisher_ab(a, b):
    return np.array([[tg(a) - tg(a + b), -tg(a + b)], [-tg(a + b), tg(b) - tg(a + b)]])

print("Beta: rho(log a, log b) | rho(logit mu, log phi) | 1-rho^2 (crate)")
for mu in [0.1, 0.3, 0.5, 0.8]:
    for phi in [1, 5, 20, 100, 1000]:
        a, b = mu * phi, (1 - mu) * phi
        I = beta_fisher_ab(a, b)
        J_log = np.diag([a, b])
        J_mp = np.array([[phi * mu * (1 - mu), a], [-phi * mu * (1 - mu), b]])
        r1, r2 = corr(J_log.T @ I @ J_log), corr(J_mp.T @ I @ J_mp)
        print(f"  mu={mu:.1f} phi={phi:5d}:  {r1:+.4f}   {r2:+.4f}   {1 - r1**2:.4f}")

# Monte-Carlo check of the analytic Beta Fisher (score outer product) at one point
rng = np.random.default_rng(0)
a, b = 3.0, 7.0
y = B(a, b).rvs(400_000, random_state=rng)
from scipy.special import digamma as dg
s = np.stack([dg(a + b) - dg(a) + np.log(y), dg(a + b) - dg(b) + np.log1p(-y)])
print("  MC check Beta(3,7):", np.round(s @ s.T / y.size, 4).tolist(), "vs", np.round(beta_fisher_ab(a, b), 4).tolist())

g = np.euler_gamma
print("\nWeibull (log lam, log k) Fisher: [[k^2, -(1-g)k], [-(1-g)k, pi^2/6+(1-g)^2]]")
print(f"  rho = {-(1 - g) / np.sqrt(np.pi**2 / 6 + (1 - g)**2):+.4f} for every (lam, k); 1-rho^2 = {1 - (1 - g)**2 / (np.pi**2 / 6 + (1 - g)**2):.4f}")
for lam, k in [(1.0, 0.7), (5.0, 2.0), (100.0, 8.0)]:
    y = W(k, scale=lam).rvs(400_000, random_state=rng)
    t = (y / lam) ** k
    s_ll = k * (t - 1)                      # d logpdf / d log lam
    s_lk = 1 + k * np.log(y / lam) * (1 - t)  # d logpdf / d log k
    eta_s_lk = s_lk + (1 - g) / k * s_ll    # score of log k at fixed eta: d log lam / d log k = (1-g)/k
    I = np.cov(np.stack([s_ll, s_lk]))
    Io = np.cov(np.stack([s_ll, eta_s_lk]))
    print(f"  MC lam={lam} k={k}: rho crate {corr(I):+.4f}  rho orthogonal {corr(Io):+.5f}")
