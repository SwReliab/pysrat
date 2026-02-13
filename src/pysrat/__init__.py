from ._version import __version__
from ._core import em_exp_emstep, sum  # noqa: F401
from .data import NHPPData, FaultData  # noqa: F401
from .emfit import EmFitResult, emfit, compare  # noqa: F401
from .nhpp import NHPPModel  # noqa: F401
from .models.exp import ExponentialNHPP, ExpSRM  # noqa: F401
from .options import nhpp_options  # noqa: F401
from .plot import plot_mvf, plot_dmvf, plot_rate, mvfplot, dmvfplot, rateplot  # noqa: F401

# backward-compatible alias (deprecated)
# backward-compatible aliases (deprecated)
faultdata = NHPPData.from_intervals
