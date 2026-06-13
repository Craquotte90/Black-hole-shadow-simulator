"""
Wrapper around scipy.integrate.solve_ivp providing:

  * High-order integration (DOP853) with strict tolerances (1e-12 by default),
    suitable for geodesic problems where conservation of E, L, and g(γ̇, γ̇)
    must hold to ~1e-10 or better.
  * Pre-built event factories: escape, capture, periastron, apastron.
  * Built-in conservation diagnostics (max relative drift of E, L, norm).

Depends on:  geodesics.py

Conventions
-----------
  * Signature (-, +, +, +).
  * Normalized units (G = c = 1).
  * State vector y = (t, r, θ, φ, ṫ, ṙ, θ̇, φ̇), shape (8,).
  * eps = -1 or eps = 0, used only for diagnostics.
"""

import warnings
import numpy as np
from scipy.integrate import solve_ivp

from geodesics import geodesic_rhs, conserved_quantities


# ---------------------------------------------------------------------------
# Event factories
# ---------------------------------------------------------------------------

def escape_event(r_max, terminal=True):
    """Event triggered when r ascends through r_max."""
    def event(lam, y):
        return y[1] - r_max
    event.terminal = terminal
    event.direction = +1
    return event


def capture_event(r_min, terminal=True):
    """Event triggered when r descends through r_min."""
    def event(lam, y):
        return y[1] - r_min
    event.terminal = terminal
    event.direction = -1
    return event


def periastron_event(terminal=False):
    """
    Event triggered at periastron passages (local minima of r).
    Uses ṙ = 0 with direction +1 (ṙ goes from negative to positive at periapse).
    """
    def event(lam, y):
        return y[5]  # ṙ
    event.terminal = terminal
    event.direction = +1
    return event


def apastron_event(terminal=False):
    """Event triggered at apastron passages (local maxima of r)."""
    def event(lam, y):
        return y[5]
    event.terminal = terminal
    event.direction = -1
    return event


# ---------------------------------------------------------------------------
# Solution container
# ---------------------------------------------------------------------------

class GeodesicSolution:
    """
    Wraps a scipy OdeSolution and adds:
      * Conservation diagnostics (E, L, norm along the trajectory).
      * Convenience accessors for state components.
    """

    def __init__(self, sol, M, eps):
        self.sol = sol
        self.M = M
        self.eps = eps

        self.t = sol.t            # affine parameter values, shape (N,)
        self.y = sol.y            # full state, shape (8, N)
        self.t_events = sol.t_events
        self.y_events = sol.y_events
        self.success = sol.success
        self.status = sol.status
        self.message = sol.message

        self._compute_diagnostics()

    def _compute_diagnostics(self):
        N = len(self.t)
        E_arr = np.empty(N)
        L_arr = np.empty(N)
        norm_arr = np.empty(N)
        for i in range(N):
            E, L, n = conserved_quantities(self.y[:, i], self.M)
            E_arr[i] = E
            L_arr[i] = L
            norm_arr[i] = n
        self.E = E_arr
        self.L = L_arr
        self.norm = norm_arr

        E0 = E_arr[0] if N > 0 else 1.0
        L0 = L_arr[0] if N > 0 else 1.0
        self.delta_E_rel_max = (np.max(np.abs(E_arr - E0)) / max(abs(E0), 1e-30)
                                if N > 0 else 0.0)
        self.delta_L_rel_max = (np.max(np.abs(L_arr - L0)) / max(abs(L0), 1e-30)
                                if N > 0 else 0.0)
        if self.eps is not None:
            self.delta_norm_max = np.max(np.abs(norm_arr - self.eps)) if N > 0 else 0.0
        else:
            n0 = norm_arr[0] if N > 0 else 0.0
            self.delta_norm_max = np.max(np.abs(norm_arr - n0)) if N > 0 else 0.0

    # ----- accessors -----
    @property
    def r(self):    return self.y[1]
    @property
    def theta(self): return self.y[2]
    @property
    def phi(self):  return self.y[3]
    @property
    def rdot(self): return self.y[5]


# ---------------------------------------------------------------------------
# Main integration wrapper
# ---------------------------------------------------------------------------

def integrate_geodesic(y0, lam_span, M,
                       eps=None,
                       events=None,
                       method='DOP853',
                       rtol=1e-12, atol=1e-12,
                       max_step=np.inf,
                       dense_output=False,
                       warn_threshold=1e-9):
    """
    Integrate a geodesic in Schwarzschild.

    Parameters
    ----------
    y0 : array, shape (8,)
        Initial state (t, r, θ, φ, ṫ, ṙ, θ̇, φ̇).
    lam_span : (float, float)
        Affine-parameter interval (lam_start, lam_end).
    M : float
        Schwarzschild mass parameter (geometric units).
    eps : float or None
        -1 for timelike, 0 for null. Used only by diagnostics.
    events : list of callables, optional
        Event functions, e.g. [capture_event(2.05), escape_event(100)].
    method : str
        ODE method passed to solve_ivp. Default 'DOP853'.
    rtol, atol : float
        Tolerances. Default 1e-12 each.
    max_step : float
        Maximum step size in λ. Default unlimited.
    dense_output : bool
        If True, store an interpolant in sol.sol.
    warn_threshold : float
        If max conservation drift exceeds this, emit a UserWarning.

    Returns
    -------
    GeodesicSolution
    """
    sol = solve_ivp(
        fun=lambda lam, y: geodesic_rhs(lam, y, M=M),
        t_span=lam_span,
        y0=y0,
        method=method,
        rtol=rtol, atol=atol,
        max_step=max_step,
        events=events,
        dense_output=dense_output,
    )
    result = GeodesicSolution(sol, M, eps)

    if (result.delta_E_rel_max > warn_threshold
            or result.delta_L_rel_max > warn_threshold
            or result.delta_norm_max > warn_threshold):
        warnings.warn(
            f"Conservation drift exceeded threshold {warn_threshold:.1e}:\n"
            f"  ΔE/E    = {result.delta_E_rel_max:.2e}\n"
            f"  ΔL/L    = {result.delta_L_rel_max:.2e}\n"
            f"  Δ|norm| = {result.delta_norm_max:.2e}\n"
            f"Consider tightening rtol/atol or reducing max_step."
        )

    return result
