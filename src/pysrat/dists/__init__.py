"""Distribution helpers (R-like naming)."""
from __future__ import annotations

from .exp import dexp, pexp, qexp, rexp
from .gamma import dgamma, pgamma, qgamma, rgamma
from .gumbel import dgumbel, pgumbel, qgumbel, rgumbel
from .lgumbel import dlgumbel, plgumbel, qlgumbel, rlgumbel
from .llogis import dllogis, pllogis, qllogis, rllogis
from .lnorm import dlnorm, plnorm, qlnorm, rlnorm
from .pareto2 import dpareto2, ppareto2, qpareto2, rpareto2
from .tgumbel import dtgumbel, ptgumbel, qtgumbel, rtgumbel
from .tlogis import dtlogis, ptlogis, qtlogis, rtlogis
from .tnorm import dtnorm, ptnorm, qtnorm, rtnorm

__all__ = [
    "dexp",
    "pexp",
    "qexp",
    "rexp",
    "dgamma",
    "pgamma",
    "qgamma",
    "rgamma",
    "dgumbel",
    "pgumbel",
    "qgumbel",
    "rgumbel",
    "dlgumbel",
    "plgumbel",
    "qlgumbel",
    "rlgumbel",
    "dllogis",
    "pllogis",
    "qllogis",
    "rllogis",
    "dlnorm",
    "plnorm",
    "qlnorm",
    "rlnorm",
    "dpareto2",
    "ppareto2",
    "qpareto2",
    "rpareto2",
    "dtgumbel",
    "ptgumbel",
    "qtgumbel",
    "rtgumbel",
    "dtlogis",
    "ptlogis",
    "qtlogis",
    "rtlogis",
    "dtnorm",
    "ptnorm",
    "qtnorm",
    "rtnorm",
]
