"""Keep small HiGHS solves single-threaded inside parallel structure trials."""
import warnings
from scipy.optimize import linprog as _linprog, OptimizeWarning


def linprog(*args,**kwargs):
    kwargs['options']=dict(kwargs.get('options') or {},threads=1)
    # SciPy passes this supported HiGHS option through, with a wrapper warning.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message=r"Unrecognized options detected: .*'threads': 1.*",category=OptimizeWarning)
        return _linprog(*args,**kwargs)
