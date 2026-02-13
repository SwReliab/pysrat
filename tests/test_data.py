from pysrat.data import NHPPData


def test_nhppdata_constructors():
    d1 = NHPPData.from_intervals(time=[1, 1, 1, 1], fault=[0, 1, 0, 5])
    assert d1.len == 4
    assert d1.total == 6

    d2 = NHPPData.from_fault_times([3, 1, 7, 15, 12], te=3)
    assert d2.len == 6

    d3 = NHPPData.from_counts([0, 1, 0, 5])
    assert d3.total == 6
