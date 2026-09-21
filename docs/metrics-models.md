# Metrics-Based NHPP Models

[`models.md`](models.md) lists the classical NHPP models, which take only the fault
counts as their input. This page covers the metrics-based (multi-factor) models, in
which software metrics enter as covariates, and explains how to add a link function of
your own.

## The model

Let $M$ be the number of faults in the software before testing starts, and let $p_k$ be
the probability that a fault not detected before is detected in the $k$-th period. If
$M$ follows a Poisson distribution with mean $\omega$, the cumulative number of detected
faults is a discrete-time NHPP with the mean value function

$$
\Lambda_n = \omega \left( 1 - \prod_{k=1}^{n} \bar p_k \right), \qquad \bar p_k = 1 - p_k .
$$

A metrics-based model makes $p_k$ depend on the covariate vector $x_k$ of the $k$-th
period, which collects the metrics observed in that period.

## Formulation as a GLM

All the models in this family are generalized linear models for $p_k$. They are
specified by a link function $g$ and an offset $o_k$:

$$
g(p_k) = \beta^{\top} x_k + o_k .
$$

| Class | $g$ | Inverse link |
|---|---|---|
| `MFLogitNHPP` | $\mathrm{logit}(p) = \log\frac{p}{1-p}$ | $1/(1+e^{-z})$ |
| `MFProbitNHPP` | $\Phi^{-1}(p)$ | $\Phi(z)$ |
| `MFCloglogNHPP` | $\mathrm{cloglog}(p) = \log(-\log(1-p))$ | $1 - e^{-e^{z}}$ |

The offset is a known term whose regression coefficient is fixed at one. It is what
makes the Cox proportional-hazards models a special case: if the model assumes

$$
p_k = 1 - (1 - p^{0}_k)^{\exp(\beta^{\top} x_k)}
$$

with baseline hazard rate $p^{0}_k$, then taking $1-p_k$ on both sides and the logarithm
twice gives

$$
\mathrm{cloglog}(p_k) = \beta^{\top} x_k + o_k, \qquad o_k = \mathrm{cloglog}(p^{0}_k) .
$$

So a Cox proportional-hazards model **is** a cloglog GLM whose offset is the cloglog of
the baseline hazard rate, whatever form that baseline takes. A constant baseline makes
$o_k$ constant, which the intercept absorbs; a discrete Weibull baseline with shape $m$
and scale $\alpha$ gives

$$
o_k = \log\left(k^{m} - (k-1)^{m}\right) - m \log \alpha ,
$$

whose second term is again absorbed by the intercept. The shape parameter $m$ alone
therefore determines the offset.

## Estimation

The parameters are estimated by an EM algorithm that treats the number of faults
remaining undetected at the end of testing as the unobserved variable. Two properties
matter for what follows.

- **The E-step does not depend on the link function.** It only computes the expected
  number of remaining faults.
- **The M-step is a fit of a GLM to weighted binomial data.** With a logit link it is an
  ordinary logistic regression.

This is why the class structure has one base class holding the EM algorithm, with each
subclass supplying only its link.

## Adding a link function

Within the M-step, the link function enters through a single step: computing the
fault-detection probability $\mu_k = g^{-1}(z_k)$ and its derivative $d\mu_k/dz_k$ from
the linear predictor $z_k$. `DynamicGLMBase` exposes exactly these as extension points.

| Method | Required | Meaning |
|---|---|---|
| `_linkinv(z)` | yes | the inverse link $g^{-1}(z)$ |
| `_dmu(z, mu)` | yes | its derivative $d\mu/dz$ |
| `_link_domain(z)` | only if restricted | boolean mask of where the link is defined |

Overriding `_dmu` is what marks a class as carrying its own link. Such a class is fitted
by the Python IRLS in `pysrat.regression._irls_binomial`; the built-in links keep going
through the C++ one. The Python implementation is a port of the C++ algorithm with the
same standardization, weights, working variate, step-halving and convergence criterion,
so the two agree to machine precision on a link they share.

### Example: logit with a Box-Cox transformation

$$
g(p; \lambda) = \frac{\{p/(1-p)\}^{\lambda} - 1}{\lambda} \quad (\lambda \neq 0),
\qquad
g(p; 0) = \log\frac{p}{1-p}
$$

```python
import numpy as np
from pysrat.nhpp.multifactor._dglm import DynamicGLMBase

EPS = 1e-12

class MFBoxCoxNHPP(DynamicGLMBase):
    link = "boxcox"

    def __init__(self, lam=0.0, has_intercept=True):
        super().__init__(has_intercept=has_intercept)
        self.lam = float(lam)

    def _linkinv(self, z):
        if self.lam == 0.0:
            return 1.0 / (1.0 + np.exp(-z))
        w = np.maximum(1.0 + self.lam * z, EPS)
        return 1.0 / (1.0 + w**(-1.0 / self.lam))

    def _dmu(self, z, mu):
        if self.lam == 0.0:
            return mu * (1.0 - mu)
        return mu * (1.0 - mu) / (1.0 + self.lam * z)

    def _link_domain(self, z):
        return 1.0 + self.lam * z > EPS
```

The inverse link requires $1 + \lambda z > 0$, which `_link_domain` declares. A step that
would leave that region gets an objective of $-\infty$, so step-halving rejects it.

### Example: generalized logit

These links are defined on the whole real line, so no `_link_domain` is needed and two
methods suffice for each.

| Model | Inverse link | $d\mu/dz$ |
|---|---|---|
| upper bound $\theta$ | $\theta\,\sigma(z)$ | $\mu(1-\mu/\theta)$ |
| Type-I generalized logistic, shape $\nu$ | $\sigma(z)^{\nu}$ | $\nu\,\mu\,(1-\mu^{1/\nu})$ |

where $\sigma$ is the logistic function.

## Link functions with a parameter

A link such as the Box-Cox one carries a shape parameter. Once that parameter is fixed,
the model is an ordinary GLM, so the EM algorithm applies unchanged. The parameter
itself is estimated by an **outer one-dimensional search**: fix the parameter, run the EM
algorithm to convergence, and compare the resulting log likelihoods.

A grid is usually safer than a derivative-free iterative method, because the log
likelihood need not be smooth in the parameter.

## Variable selection

`stepwise` adds and removes covariates until AIC stops improving. It works the same way
whatever the link function.

```python
from pysrat.nhpp import stepwise

best = stepwise(model, direction="both")   # "forward" / "backward" also available
print(best.aic_, best.data_.metrics_name)
```

The intercept is kept by default (`keep_intercept=True`). When the number of covariates
is small, fitting every subset and comparing AIC directly is an alternative that avoids
the greediness of the stepwise search.

## API reference

| Name | Where | Purpose |
|---|---|---|
| `DMetricsData` | `pysrat.data` | fault counts, covariate matrix and offset |
| `DMetricsData.from_dataframe` | `pysrat.data` | build it from a `pandas.DataFrame` |
| `MFLogitNHPP` | `pysrat.nhpp` | logit link |
| `MFProbitNHPP` | `pysrat.nhpp` | probit link |
| `MFCloglogNHPP` | `pysrat.nhpp` | cloglog link |
| `DynamicGLMBase` | `pysrat.nhpp.multifactor._dglm` | base class; extension points |
| `stepwise` | `pysrat.nhpp` | AIC-based variable selection |
| `plot_dmvf` | `pysrat.nhpp` | plot the fitted mean value function |
| `glm_binomial` | `pysrat.regression` | the binomial GLM fit used by the M-step |

`from_dataframe` takes `metrics`, `fault` and an optional `offset`, each naming columns
of the data frame. **Do not include a constant column in `metrics`**: the intercept is
handled by the model through `has_intercept`, and passing `intercept=True` raises
`ValueError`.
