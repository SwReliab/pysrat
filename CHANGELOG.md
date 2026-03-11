# pysrat 0.2.3

- add `offset` argument to `fit_pr_nhpp` to override `SMetricsData.offset` at call time
- add PR-NHPP tests covering offset override, `SMetricsData.offset` default usage, and offset shape validation
- update `README.md` to use `fit_pr_nhpp` naming and document offset usage in PR-NHPP examples

# pysrat 0.2.2

- add `total_time` property to `NHPPData` for convenience in accessing total observed time
- add `total_count` property to `NHPPData` for convenience in accessing total observed count

# pysrat 0.2.1

- add `truncate` method to `NHPPData` for truncating data at a specified time

