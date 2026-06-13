"""
Ray-tracing of null geodesics in Schwarzschild to produce an image of a
black-hole shadow as seen by a static observer at finite distance r_obs.

Pipeline
--------
  1. Build the static observer's orthonormal tetrad at (r_obs, π/2).
  2. For each pixel (i, j):
       a. Compute camera angles (β, γ).
       b. Build the unit spatial direction n̂ in the local frame.
       c. Build the null 4-vector k in coordinate components.
       d. Verify g(k, k) = 0 (catches tetrad bugs).
       e. Integrate the null geodesic from the camera outward.
  3. Color the pixel:
       * black if captured (r < r_capture);
       * checkerboard color if escaped (using asymptotic (θ_∞, φ_∞));
       * grey if neither event triggered (rare).
  4. Validate against the analytical shadow angle
        sin α_shadow = (3√3 M / r_obs) √(1 - 2M/r_obs).

Run as a script:  python shadow.py
"""

import time
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt

from integrator import integrate_geodesic, escape_event, capture_event


# ---------------------------------------------------------------------------
# Tetrad and ray construction
# ---------------------------------------------------------------------------

def static_tetrad(r_obs, theta_obs, M):
    """
    Orthonormal tetrad (e_(t), e_(r), e_(θ), e_(φ)) for a static observer.

    Returns
    -------
    e_coord : ndarray, shape (4, 4)
        e_coord[a, μ] = μ-component of the tetrad vector e_(a).
        e.g. e_coord[0, 0] = (1-2M/r)^(-1/2)  → e_(t)^t component.
    """
    f = 1.0 - 2.0 * M / r_obs
    sqrt_f = np.sqrt(f)
    sin_t = np.sin(theta_obs)
    e = np.zeros((4, 4))
    e[0, 0] = 1.0 / sqrt_f
    e[1, 1] = sqrt_f
    e[2, 2] = 1.0 / r_obs
    e[3, 3] = 1.0 / (r_obs * sin_t)
    return e


def pixel_to_angles(i, j, N, fov):
    """
    Convert pixel (i, j) to camera angles (β, γ).

    i : row index, 0 = top of image.
    j : column index, 0 = left of image.
    N : image side length.
    fov : full field of view in radians.
    """
    beta  = fov * ((j + 0.5) / N - 0.5)
    gamma = -fov * ((i + 0.5) / N - 0.5)
    return beta, gamma


def angles_to_nhat(beta, gamma):
    """
    Tetrad components of the unit spatial direction n̂ at the camera:

        n̂ = -cos(γ)cos(β) e_(r)
             + sin(γ)        e_(θ)
             + cos(γ)sin(β)  e_(φ)

    Returns (n_r, n_θ, n_φ).
    """
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    return -cg * cb, sg, cg * sb


def initial_4velocity(r_obs, theta_obs, beta, gamma, M):
    """
    Coordinate components (k^t, k^r, k^θ, k^φ) of the null tangent at
    the camera, for a ray entering pixel (β, γ).

    Built as k = e_(t) + n̂ in tetrad form, then expanded over the
    coordinate basis using the tetrad of `static_tetrad`.
    """
    f = 1.0 - 2.0 * M / r_obs
    sqrt_f = np.sqrt(f)
    sin_t = np.sin(theta_obs)
    nr, nth, nph = angles_to_nhat(beta, gamma)
    kt = 1.0 / sqrt_f
    kr = nr * sqrt_f
    kth = nth / r_obs
    kph = nph / (r_obs * sin_t)
    return np.array([kt, kr, kth, kph])


def null_check(y, r_obs, theta_obs, M):
    """
    Compute g(k, k) at the camera for state y. Should be < 1e-13 for a
    correctly built ray.
    """
    f = 1.0 - 2.0 * M / r_obs
    sin_t = np.sin(theta_obs)
    kt, kr, kth, kph = y[4:]
    return (-f * kt**2
            + kr**2 / f
            + r_obs**2 * kth**2
            + r_obs**2 * sin_t**2 * kph**2)


# ---------------------------------------------------------------------------
# Sky map (spherical checkerboard)
# ---------------------------------------------------------------------------

# Two cell colors. Pure black is reserved for the BH shadow itself.
SKY_LIGHT = (0.95, 0.95, 0.90)
SKY_DARK  = (0.25, 0.30, 0.55)


def checkerboard_color(theta, phi, n_theta=10, n_phi=20):
    """
    Spherical checkerboard. Returns an RGB triplet in [0, 1].
    n_phi = 2 * n_theta gives roughly square cells at the equator.
    """
    phi = phi % (2.0 * np.pi)
    i_th = int(np.floor(n_theta * theta / np.pi))
    i_ph = int(np.floor(n_phi * phi / (2.0 * np.pi)))
    return SKY_LIGHT if (i_th + i_ph) % 2 == 0 else SKY_DARK


# ---------------------------------------------------------------------------
# Single-ray trace
# ---------------------------------------------------------------------------

COLOR_CAPTURED      = (0.0, 0.0, 0.0)
COLOR_UNDETERMINED  = (0.5, 0.5, 0.5)


def trace_pixel(i, j, N, fov, r_obs, theta_obs, M,
                r_capture=2.05, r_escape=100.0,
                lam_max=500.0, max_step=0.5,
                n_theta=10, n_phi=20,
                sky_fn=None):
    """
    Trace a single pixel and return its RGB color.

    sky_fn : callable(theta, phi) -> (r, g, b), optional
        Custom celestial-sphere color function. If None (default),
        the spherical checkerboard with (n_theta, n_phi) cells is used.
        Must be a module-level function for ProcessPoolExecutor pickling.
    """
    beta, gamma = pixel_to_angles(i, j, N, fov)
    k = initial_4velocity(r_obs, theta_obs, beta, gamma, M)
    y0 = np.array([0.0, r_obs, theta_obs, 0.0, k[0], k[1], k[2], k[3]])

    if abs(null_check(y0, r_obs, theta_obs, M)) > 1e-10:
        return COLOR_UNDETERMINED

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
        return COLOR_CAPTURED

    if len(escaped) > 0:
        y_end = sol.y_events[1][0]
        if sky_fn is not None:
            return sky_fn(y_end[2], y_end[3])
        return checkerboard_color(y_end[2], y_end[3], n_theta, n_phi)

    return COLOR_UNDETERMINED


# ---------------------------------------------------------------------------
# Full-image renderer
# ---------------------------------------------------------------------------

def _trace_row(i, N, fov, r_obs, theta_obs, M,
               r_capture, r_escape, lam_max, max_step, n_theta, n_phi,
               sky_fn=None):
    """Trace one row (used by ProcessPoolExecutor)."""
    row = np.empty((N, 3))
    for j in range(N):
        row[j] = trace_pixel(i, j, N, fov, r_obs, theta_obs, M,
                             r_capture, r_escape, lam_max, max_step,
                             n_theta, n_phi,
                             sky_fn=sky_fn)
    return row


def render_image(N, fov, r_obs, theta_obs, M,
                 r_capture=2.05, r_escape=100.0,
                 lam_max=500.0, max_step=0.5,
                 n_theta=10, n_phi=20,
                 sky_fn=None,
                 parallel=True, n_workers=None,
                 verbose=True):
    """
    Render an N × N image of the BH shadow.

    sky_fn : callable(theta, phi) -> (r, g, b), optional
        Custom celestial-sphere color function. If None, uses the
        spherical checkerboard. Must be a top-level (importable)
        function for ProcessPoolExecutor pickling.

    Returns
    -------
    image : ndarray, shape (N, N, 3), values in [0, 1].
    """
    t0 = time.time()
    if parallel:
        worker = partial(_trace_row,
                         N=N, fov=fov,
                         r_obs=r_obs, theta_obs=theta_obs, M=M,
                         r_capture=r_capture, r_escape=r_escape,
                         lam_max=lam_max, max_step=max_step,
                         n_theta=n_theta, n_phi=n_phi,
                         sky_fn=sky_fn)
        with ProcessPoolExecutor(max_workers=n_workers) as exe:
            rows = list(exe.map(worker, range(N)))
        image = np.array(rows)
    else:
        image = np.empty((N, N, 3))
        for i in range(N):
            if verbose and i % max(1, N // 20) == 0:
                print(f"  row {i}/{N}  ({time.time()-t0:.1f}s elapsed)")
            for j in range(N):
                image[i, j] = trace_pixel(i, j, N, fov, r_obs, theta_obs, M,
                                          r_capture, r_escape,
                                          lam_max, max_step,
                                          n_theta, n_phi,
                                          sky_fn=sky_fn)
    if verbose:
        print(f"  render done in {time.time()-t0:.1f}s "
              f"({N*N} rays, {(time.time()-t0)/(N*N)*1e3:.2f} ms/ray)")
    return image


# ---------------------------------------------------------------------------
# Validation against the analytical shadow size
# ---------------------------------------------------------------------------

def predicted_shadow_angle(r_obs, M):
    """
    Analytical shadow angular radius:

        sin α_shadow = (3√3 M / r_obs) √(1 - 2M/r_obs).
    """
    x = 3.0 * np.sqrt(3.0) * M / r_obs * np.sqrt(1.0 - 2.0 * M / r_obs)
    return np.arcsin(x)


def measure_shadow_radius(image, fov):
    """
    Measure the shadow's angular radius by scanning the central row
    and locating the black-to-not-black transitions.

    Returns the measured α (radians), or None if no shadow detected.
    """
    N = image.shape[0]
    center_row = image[N // 2]
    is_black = center_row.sum(axis=1) < 0.05
    if not np.any(is_black):
        return None
    idx = np.where(is_black)[0]
    j_min, j_max = idx[0], idx[-1]
    half_width_pix = max(N / 2 - j_min, j_max - N / 2)
    return fov * half_width_pix / N


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def show_image(image, title=None, overlay_theoretical=None, fov=None):
    """
    Display the rendered image; optionally overlay the predicted shadow circle.

    overlay_theoretical : float or None
        Predicted shadow angle (radians) to draw as a dashed circle.
    fov : float
        Required if overlay_theoretical is given (sets the angular scale).
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image, origin='upper', extent=[-1, 1, -1, 1])
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title)
    if overlay_theoretical is not None and fov is not None:
        r_norm = overlay_theoretical / (fov / 2.0)
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(r_norm * np.cos(th), r_norm * np.sin(th),
                'r--', lw=1.5, alpha=0.85, label='theoretical shadow')
        ax.legend(loc='lower right', facecolor='white', framealpha=0.8)
    return fig, ax


# ---------------------------------------------------------------------------
# Main: incremental validation pipeline
# ---------------------------------------------------------------------------
"""
def main():
    M         = 1.0
    r_obs     = 30.0 * M
    theta_obs = np.pi / 2
    fov       = np.deg2rad(30.0)

    print("=" * 60)
    print("STEP 1 — Pixel-level sanity (g(k, k) must vanish)")
    print("=" * 60)
    for (i, j, name) in [(50, 50, "center"),
                         (0, 0, "top-left"),
                         (99, 99, "bottom-right"),
                         (50, 0, "middle-left"),
                         (0, 50, "top-middle")]:
        b, g = pixel_to_angles(i, j, 100, fov)
        k = initial_4velocity(r_obs, theta_obs, b, g, M)
        y0 = np.array([0, r_obs, theta_obs, 0, *k])
        n2 = null_check(y0, r_obs, theta_obs, M)
        status = "OK" if abs(n2) < 1e-13 else "** FAIL **"
        print(f"  pixel {name:15s}: β = {b:+.3f}, γ = {g:+.3f}, "
              f"g(k,k) = {n2:+.2e}  [{status}]")

    print()
    print("=" * 60)
    print("STEP 2 — Analytical prediction")
    print("=" * 60)
    alpha_pred = predicted_shadow_angle(r_obs, M)
    print(f"  Camera position   : r_obs = {r_obs/M:.1f} M, θ_obs = π/2")
    print(f"  Field of view     : {np.rad2deg(fov):.2f}°")
    print(f"  Predicted α_shadow: {np.rad2deg(alpha_pred):.3f}°"
          f"   (sin α = {np.sin(alpha_pred):.4f})")
    print(f"  Shadow / half-FOV : {alpha_pred / (fov/2):.3f}")

    print()
    print("=" * 60)
    print("STEP 3 — Low-resolution render (100 × 100)")
    print("=" * 60)
    img_low = render_image(
        N=100, fov=fov, r_obs=r_obs, theta_obs=theta_obs, M=M,
        parallel=True, verbose=True,
    )
    alpha_meas = measure_shadow_radius(img_low, fov)
    if alpha_meas is not None:
        err = abs(alpha_meas - alpha_pred) / alpha_pred * 100
        print(f"  Measured α_shadow : {np.rad2deg(alpha_meas):.3f}°")
        print(f"  Relative error    : {err:.2f} %")
    else:
        print("  No shadow detected — check r_capture, fov, or integration.")

    fig, ax = show_image(
        img_low,
        title=f"BH shadow (100×100, r_obs={r_obs/M:.0f} M, FOV={np.rad2deg(fov):.0f}°)",
        overlay_theoretical=alpha_pred, fov=fov,
    )
    fig.savefig("shadow_lowres.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: shadow_lowres.png")
"""
