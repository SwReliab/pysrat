# pysrat 0.2.7

## Added

- User-defined link functions for the multi-factor (metrics-based) NHPP models.
  A subclass of `DynamicGLMBase` that overrides `_dmu` (the derivative of the
  inverse link) supplies its own link, and may declare a restricted domain with
  `_link_domain`. Nothing else has to be written: the E-step, the M-step, the
  likelihood and AIC all come from the base class.
- `glm_binomial` accepts a tuple `(linkinv, dmu)` or `(linkinv, dmu, domain)`
  in place of a link name, and fits it with `regression._irls_binomial`, a
  Python port of the C++ IRLS with the same standardization, weights, working
  variate, step-halving and convergence criterion.

## Unchanged

- The built-in links (`logit`, `probit`, `cloglog`) still go through the C++
  IRLS. Their results and their speed are untouched.

# pysrat 0.2.6

- refactor binomial regression wrappers (`glm_binomial`, `glmnet_binomial`) and align C++/pybind integration
- add Python-side validation for binomial response (`y`/`n_trials`) in binomial GLM/GLMNET wrappers
- simplify `NHPPData` constructors to use `from_intervals` as the base path and remove `from_event_times`

# pysrat 0.2.5

- refactor Poisson regression wrappers (`glm_poisson` and `glmnet_poisson`) and corresponding C++/pybind integration
- update `fit_pr_nhpp` to use penalty factorization (`penalty`) in regression forwarding
- add `l2matrix` support to `fit_pr_nhpp` and pass it through to Poisson GLM/GLMNET backends

# pysrat 0.2.4

- fix tests to use current `NHPPData.from_intervals` keyword arguments (`intervals`, `counts`, `on_boundary`) instead of removed legacy names (`time`, `fault`, `type`)
- replace removed `NHPPData.from_counts` and `NHPPData.from_fault_times` calls in tests with `from_intervals`
- add dedicated call for Poisson regression with `alpha=0` (L2-only / ridge)

# pysrat 0.2.3

- add `offset` argument to `fit_pr_nhpp` to override `SMetricsData.offset` at call time
- add PR-NHPP tests covering offset override, `SMetricsData.offset` default usage, and offset shape validation
- update `README.md` to use `fit_pr_nhpp` naming and document offset usage in PR-NHPP examples

# pysrat 0.2.2

- add `total_time` property to `NHPPData` for convenience in accessing total observed time
- add `total_count` property to `NHPPData` for convenience in accessing total observed count

# pysrat 0.2.1

- add `truncate` method to `NHPPData` for truncating data at a specified time

