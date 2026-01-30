import enum
import numpy as np
import torch
from xitorch.interpolate.interp1 import Interp1D


class SplineType(enum.Enum):
    ZERO = enum.auto()
    LINEAR = enum.auto()
    QUADRATIC = enum.auto()
    CUBIC = enum.auto()


class TorchSpline:
    """Drop-in replacement for the interp1d object returned by make_spline.

    controls: (..., T, m), times: (T,). Query t of any shape -> (..., *t.shape, m).
    """

    def __init__(self, times, controls, spline_order: SplineType = SplineType.ZERO, extrapolate: bool = True):
        self._t = torch.as_tensor(times, dtype=torch.float64)
        self._y = torch.as_tensor(controls, dtype=torch.float64).transpose(-1, -2)  # (..., m, T)
        self._t0, self._t1 = float(self._t[0]), float(self._t[-1])
        self._order = spline_order

        if spline_order == SplineType.LINEAR:
            self._interp = Interp1D(self._t, self._y, method="linear",
                                    extrap="bound" if extrapolate else "periodic")
        elif spline_order == SplineType.CUBIC:
            self._interp = Interp1D(self._t, self._y, method="cspline", bc_type="not-a-knot",
                                    extrap="bound" if extrapolate else "periodic")
        elif spline_order == SplineType.ZERO:
            self._interp = None  # handled in __call__
        else:
            raise ValueError(f"Unsupported spline_order '{spline_order}'")

    def __call__(self, query) -> torch.Tensor:
        xq = torch.as_tensor(query, dtype=torch.float64)
        # Clamping == fill_value=(first, last) with bounds_error=False.
        xq = xq.reshape(-1).clamp(self._t0, self._t1)

        if self._order == SplineType.ZERO:
            idx = torch.searchsorted(self._t, xq, right=True) - 1
            idx = idx.clamp(0, len(self._t) - 1)
            yq = self._y[..., idx]  # (..., m, T)
        else:
            yq = self._interp(xq)  # (..., m, T)

        return yq.transpose(-1, -2)  # (..., T, m)
