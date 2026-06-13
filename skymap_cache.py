"""
Compute and cache the pixel → sky map for a Schwarzschild ray-tracing camera.

Null geodesic integration (the expensive part) is performed only ONCE per
camera configuration and stored on disk. All subsequent rendering operations
(still image, video, background swap) become near-free lookups.

Cache format:
  - <name>.npz : compressed NumPy arrays
      * theta_inf[i, j] : θ direction at infinity (float32)
      * phi_inf[i, j]   : φ direction at infinity (float32)
      * captured[i, j]  : bool, true if the ray fell into the horizon
  - <name>.json : human-readable metadata (physical + integrator parameters)
"""

import json
import os
import time
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from shadow import pixel_to_angles, initial_4velocity, null_check
from integrator import integrate_geodesic, capture_event, escape_event


DEFAULTS = {
    'r_capture': 2.05,
    'r_escape':  100.0,
    'lam_max':   500.0,
    'max_step':  0.5,
    'rtol':      1e-12,
    'atol':      1e-12,
}


# ---------------------------------------------------------------------------
# Ray tracing: variant returning asymptotic angles instead of a color
# ---------------------------------------------------------------------------

def trace_pixel_skymap(i, j, N, fov, r_obs, theta_obs, M,
                       r_capture, r_escape, lam_max, max_step):
    """
    Trace a single ray and return (theta_inf, phi_inf, captured).

    captured = True if the ray ended in the horizon (or remained undetermined).
    Otherwise (theta_inf, phi_inf) is the asymptotic direction on the
    celestial sphere, suitable for texture lookup.
    """
    beta, gamma = pixel_to_angles(i, j, N, fov)
    k = initial_4velocity(r_obs, theta_obs, beta, gamma, M)
    y0 = np.array([0.0, r_obs, theta_obs, 0.0, k[0], k[1], k[2], k[3]])

    if abs(null_check(y0, r_obs, theta_obs, M)) > 1e-10:
        return (np.nan, np.nan, True)

    sol = integrate_geodesic(
        y0=y0,
        lam_span=(0.0, lam_max),
        M=M, eps=0.0,
        events=[capture_event(r_capture), escape_event(r_escape)],
        max_step=max_step,
        warn_threshold=np.inf,
    )

    captured = sol.t_events[0]
    escaped  = sol.t_events[1]

    if len(captured) > 0 and (len(escaped) == 0 or captured[0] < escaped[0]):
        return (np.nan, np.nan, True)
    if len(escaped) > 0:
        y_end = sol.y_events[1][0]
        return (y_end[2], y_end[3], False)
    return (np.nan, np.nan, True)


def _trace_row(i, N, fov, r_obs, theta_obs, M,
               r_capture, r_escape, lam_max, max_step):
    """Trace one row (used by ProcessPoolExecutor)."""
    row_th  = np.empty(N)
    row_ph  = np.empty(N)
    row_cap = np.empty(N, dtype=bool)
    for j in range(N):
        th, ph, cap = trace_pixel_skymap(
            i, j, N, fov, r_obs, theta_obs, M,
            r_capture, r_escape, lam_max, max_step,
        )
        row_th[j], row_ph[j], row_cap[j] = th, ph, cap
    return row_th, row_ph, row_cap


# ---------------------------------------------------------------------------
# Full skymap computation
# ---------------------------------------------------------------------------

def compute_skymap(N, fov, r_obs, theta_obs, M,
                   r_capture=DEFAULTS['r_capture'],
                   r_escape=DEFAULTS['r_escape'],
                   lam_max=DEFAULTS['lam_max'],
                   max_step=DEFAULTS['max_step'],
                   parallel=True, n_workers=None, verbose=True):
    """
    Compute the pixel → sky map for the given parameters.

    Returns a dict with keys 'theta_inf', 'phi_inf', 'captured', each an
    (N, N) array. theta_inf and phi_inf are float32; captured rays carry
    NaN in those two arrays.
    """
    t0 = time.time()

    if parallel:
        worker = partial(_trace_row,
                         N=N, fov=fov,
                         r_obs=r_obs, theta_obs=theta_obs, M=M,
                         r_capture=r_capture, r_escape=r_escape,
                         lam_max=lam_max, max_step=max_step)
        with ProcessPoolExecutor(max_workers=n_workers) as exe:
            rows = list(exe.map(worker, range(N)))
        theta_inf = np.array([r[0] for r in rows])
        phi_inf   = np.array([r[1] for r in rows])
        captured  = np.array([r[2] for r in rows])
    else:
        theta_inf = np.empty((N, N))
        phi_inf   = np.empty((N, N))
        captured  = np.empty((N, N), dtype=bool)
        for i in range(N):
            if verbose and i % max(1, N // 20) == 0:
                print(f"  row {i}/{N}  ({time.time()-t0:.1f}s)")
            for j in range(N):
                th, ph, cap = trace_pixel_skymap(
                    i, j, N, fov, r_obs, theta_obs, M,
                    r_capture, r_escape, lam_max, max_step,
                )
                theta_inf[i, j], phi_inf[i, j], captured[i, j] = th, ph, cap

    if verbose:
        n_captured = int(captured.sum())
        print(f"  done in {time.time()-t0:.1f}s  "
              f"({n_captured}/{N*N} pixels capturés, "
              f"soit {100*n_captured/N**2:.1f}%)")

    return {
        'theta_inf': theta_inf.astype(np.float32),
        'phi_inf':   phi_inf.astype(np.float32),
        'captured':  captured,
    }


# ---------------------------------------------------------------------------
# Flat-space skymap: no black hole, for pedagogical comparison
# ---------------------------------------------------------------------------

def flat_skymap(N, fov, theta_obs=np.pi/2):
    """
    Build a skymap corresponding to flat space (no black hole).

    For each pixel (i, j), the asymptotic direction is computed by
    geometric projection.

    Returns
    -------
    skymap : dict with the same keys as compute_skymap.
        captured is all False (nothing is captured in flat space).
    """
    j_idx, i_idx = np.meshgrid(np.arange(N), np.arange(N), indexing='xy')
    beta  = fov * ((j_idx + 0.5) / N - 0.5)
    gamma = -fov * ((i_idx + 0.5) / N - 0.5)

    cg, sg = np.cos(gamma), np.sin(gamma)
    cb, sb = np.cos(beta),  np.sin(beta)

    # Cartesian ray direction in camera frame:
    # camera at +x, optical axis toward -x; e_θ = +z ("up"), e_φ = +y ("right").
    nx = -cg * cb
    ny =  cg * sb
    nz =  sg

    theta_inf = np.arccos(np.clip(nz, -1.0, 1.0)).astype(np.float32)
    phi_inf   = (np.arctan2(ny, nx) % (2.0 * np.pi)).astype(np.float32)
    captured  = np.zeros((N, N), dtype=bool)

    return {
        'theta_inf': theta_inf,
        'phi_inf':   phi_inf,
        'captured':  captured,
    }


# ---------------------------------------------------------------------------
# Deterministic naming and I/O
# ---------------------------------------------------------------------------

def skymap_filename(r_obs, theta_obs, N, fov, M=1.0,
                    prefix='skymap', cache_dir='cache'):
    """
    Build two paths (npz, json) from the camera parameters.
    """
    name = (f"{prefix}"
            f"_robs{r_obs/M:.0f}M"
            f"_theta{np.rad2deg(theta_obs):.0f}deg"
            f"_N{N}"
            f"_fov{np.rad2deg(fov):.0f}deg")
    base = Path(cache_dir) / name
    return str(base) + '.npz', str(base) + '.json'


def save_skymap(skymap, npz_path, json_path, params):
    """Save the three arrays in compressed .npz plus a JSON sidecar."""
    np.savez_compressed(
        npz_path,
        theta_inf=skymap['theta_inf'],
        phi_inf=skymap['phi_inf'],
        captured=skymap['captured'],
    )
    with open(json_path, 'w') as f:
        json.dump(params, f, indent=2)


def load_skymap(npz_path, json_path):
    """Load a skymap from disk. Returns (skymap_dict, params_dict)."""
    with np.load(npz_path) as data:
        skymap = {
            'theta_inf': data['theta_inf'].astype(np.float64),
            'phi_inf':   data['phi_inf'].astype(np.float64),
            'captured':  data['captured'],
        }
    with open(json_path) as f:
        params = json.load(f)
    return skymap, params


# ---------------------------------------------------------------------------
# Main API: "compute if needed, else load"
# ---------------------------------------------------------------------------

def get_or_compute_skymap(N, fov, r_obs,
                          theta_obs=np.pi/2, M=1.0,
                          cache_dir='cache',
                          force_recompute=False,
                          parallel=True, n_workers=None, verbose=True,
                          **integrator_kwargs):
    """
    Load the skymap if the file exists, otherwise compute and save it.

    Parameters
    ----------
    N : int
        Image resolution (N × N pixels).
    fov : float
        Field of view in radians.
    r_obs : float
        Camera distance to the black hole (in units of M).
    theta_obs : float
        Camera latitude (default π/2, equatorial plane).
    M : float
        Mass in geometric units (default 1).
    cache_dir : str
        Directory where cache files are stored.
    force_recompute : bool
        If True, ignore the cache and recompute.
    parallel : bool
        Enable parallelization via ProcessPoolExecutor.
    **integrator_kwargs : optional
        r_capture, r_escape, lam_max, max_step, etc. — see DEFAULTS.

    Returns
    -------
    skymap : dict with keys 'theta_inf', 'phi_inf', 'captured'.
    """
    os.makedirs(cache_dir, exist_ok=True)
    npz_path, json_path = skymap_filename(
        r_obs, theta_obs, N, fov, M, cache_dir=cache_dir,
    )

    if (not force_recompute
            and os.path.exists(npz_path)
            and os.path.exists(json_path)):
        if verbose:
            print(f"Loading cached skymap from {npz_path}")
        skymap, _ = load_skymap(npz_path, json_path)
        return skymap

    if verbose:
        print(f"Computing skymap : r_obs={r_obs/M:.0f}M, N={N}, "
              f"fov={np.rad2deg(fov):.0f}°, theta_obs={np.rad2deg(theta_obs):.0f}°")

    int_params = {**{k: DEFAULTS[k] for k in
                     ('r_capture', 'r_escape', 'lam_max', 'max_step')},
                  **integrator_kwargs}

    skymap = compute_skymap(
        N=N, fov=fov, r_obs=r_obs, theta_obs=theta_obs, M=M,
        r_capture=int_params['r_capture'],
        r_escape=int_params['r_escape'],
        lam_max=int_params['lam_max'],
        max_step=int_params['max_step'],
        parallel=parallel, n_workers=n_workers, verbose=verbose,
    )

    params = {
        'r_obs': float(r_obs),
        'theta_obs': float(theta_obs),
        'N': int(N),
        'fov': float(fov),
        'M': float(M),
        **int_params,
    }
    save_skymap(skymap, npz_path, json_path, params)
    if verbose:
        size_kb = os.path.getsize(npz_path) / 1024
        print(f"Saved to {npz_path}  ({size_kb:.1f} KB)")

    return skymap
