"""
Static image rendering from a precomputed skymap.

A render consists of a lookup on the theta_inf / phi_inf map stored in the
skymap, plus the application of a background (equirectangular texture or
sky_fn callable) and optionally some foreground overlays.

No geodesic integration is performed here.
"""

import numpy as np


COLOR_CAPTURED = np.array([0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Background lookup
# ---------------------------------------------------------------------------

def sample_background(skymap, background):
    """
    Evaluate the background on all non-captured pixels of the skymap.

    background can be:
      - an ndarray (H, W, 3) — equirectangular texture;
      - a callable (theta, phi) -> (r, g, b) — sky function.

    For captured pixels (NaN in theta_inf), returns a placeholder color
    that will be overwritten by COLOR_CAPTURED in render().
    """
    theta_inf = skymap['theta_inf']
    phi_inf   = skymap['phi_inf']
    captured  = skymap['captured']

    if isinstance(background, np.ndarray):
        return _sample_texture_vec(background, theta_inf, phi_inf, captured)

    H, W = theta_inf.shape
    out = np.empty((H, W, 3))
    for i in range(H):
        for j in range(W):
            if captured[i, j]:
                out[i, j] = 0.0
            else:
                out[i, j] = background(float(theta_inf[i, j]),
                                       float(phi_inf[i, j]))
    return out


def _sample_texture_vec(texture, theta_inf, phi_inf, captured):
    """
    Vectorized bilinear lookup on an equirectangular texture.

    texture : (H, W, 3), float ∈ [0, 1]. Convention:
       - axis 0 (rows)    : θ ∈ [0, π], 0 at top, π at bottom.
       - axis 1 (columns) : φ ∈ [0, 2π], 0 at left, 2π at right.
    """
    H, W = texture.shape[:2]
    th = np.where(captured, 0.0, theta_inf)
    ph = np.where(captured, 0.0, phi_inf) % (2.0 * np.pi)

    y = (th / np.pi) * H
    x = (ph / (2.0 * np.pi)) * W

    y0 = np.floor(y).astype(int) % H
    x0 = np.floor(x).astype(int) % W
    y1 = (y0 + 1) % H
    x1 = (x0 + 1) % W
    fy = (y - np.floor(y))[..., None]
    fx = (x - np.floor(x))[..., None]

    c00 = texture[y0, x0]
    c01 = texture[y0, x1]
    c10 = texture[y1, x0]
    c11 = texture[y1, x1]

    return ((1 - fy) * (1 - fx) * c00
          + (1 - fy) * fx       * c01
          + fy       * (1 - fx) * c10
          + fy       * fx       * c11)


# ---------------------------------------------------------------------------
# Angular distance and overlay mask
# ---------------------------------------------------------------------------

def overlay_mask(skymap, theta_obj, phi_obj, radius_rad):
    """
    Boolean mask of pixels whose asymptotic direction lies within an
    angular radius `radius_rad` of (theta_obj, phi_obj), excluding
    captured pixels.

    Great-circle angular distance:
       cos(d) = cos(θ₁)cos(θ₂) + sin(θ₁)sin(θ₂)cos(φ₁-φ₂).
    """
    theta_inf = skymap['theta_inf']
    phi_inf   = skymap['phi_inf']
    captured  = skymap['captured']

    cos_d = (np.cos(theta_inf) * np.cos(theta_obj)
             + np.sin(theta_inf) * np.sin(theta_obj)
               * np.cos(phi_inf - phi_obj))
    cos_d = np.clip(cos_d, -1.0, 1.0)
    d = np.arccos(cos_d)
    return (d < radius_rad) & (~captured)


# ---------------------------------------------------------------------------
# Full image rendering
# ---------------------------------------------------------------------------

def render(skymap, background, overlays=None,
           capture_color=COLOR_CAPTURED):
    """
    Compose an (N, N, 3) image from a skymap, a background, and
    optional overlays.

    Parameters
    ----------
    skymap : dict
        Output of get_or_compute_skymap.
    background : ndarray (H, W, 3) or callable
        Celestial-sphere source.
    overlays : list of dict, optional
        Each overlay: {'theta', 'phi', 'color', 'radius_rad'}.
        Pixels within the angular neighborhood of (theta, phi) are
        painted with `color`, overriding the background.
    capture_color : array-like (3,)
        Color of captured pixels (default black).

    Returns
    -------
    image : ndarray (N, N, 3), values in [0, 1].
    """
    image = sample_background(skymap, background)
    image[skymap['captured']] = capture_color

    if overlays:
        for ov in overlays:
            mask = overlay_mask(skymap, ov['theta'], ov['phi'], ov['radius_rad'])
            image[mask] = np.asarray(ov['color'], dtype=image.dtype)

    return np.clip(image, 0, 1)


# ---------------------------------------------------------------------------
# Texture loading helper
# ---------------------------------------------------------------------------

def load_texture(filepath, target_size=(1024, 512)):
    """
    Load an equirectangular image from disk and resize it.

    Expected format: width ≈ 2 × height (celestial sphere ratio).
    Requires Pillow.
    """
    from PIL import Image
    W, H = target_size
    img = Image.open(filepath).convert('RGB')
    img = img.resize((W, H), Image.LANCZOS)
    return np.asarray(img, dtype=np.float64) / 255.0
