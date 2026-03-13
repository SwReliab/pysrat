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

