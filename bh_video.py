"""
Video generation for moving objects passing behind a black hole.

Reuses bh_render.render() frame by frame.

Supported output formats:
  - .mp4    via FFMpegWriter (requires ffmpeg on the system)
  - .gif    via PillowWriter (always available)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

from bh_render import render


# ---------------------------------------------------------------------------
# Standard trajectories on the celestial sphere
# ---------------------------------------------------------------------------

def linear_sky_trajectory(theta0, phi0, theta1, phi1):
    """
    Linear trajectory on the celestial sphere: linear interpolation
    between (theta0, phi0) and (theta1, phi1) for t ∈ [0, 1].

    To visualize gravitational lensing, choose a trajectory passing
    near φ = 0 (the antipode of the optical axis). Example:
        traj = linear_sky_trajectory(np.pi/2, -0.4, np.pi/2, +0.4)
    makes an object cross through the Einstein ring.
    """
    def f(t):
        return (theta0 + t * (theta1 - theta0),
                phi0 + t * (phi1 - phi0))
    return f


def circular_sky_trajectory(theta_center, phi_center, radius_rad, n_turns=1):
    """
    Circular trajectory centered on (theta_center, phi_center),
    of angular radius `radius_rad`, traversed `n_turns` times for t ∈ [0, 1].

    Small-circle in local planar projection (valid for radius_rad ≲ 0.2 rad).
    """
    def f(t):
        angle = 2 * np.pi * n_turns * t
        return (theta_center + radius_rad * np.cos(angle),
                phi_center   + radius_rad * np.sin(angle))
    return f


# ---------------------------------------------------------------------------
# Video generation
# ---------------------------------------------------------------------------

def make_video(skymap, background, trajectory,
               n_frames=100, fps=20,
               object_color=(1.0, 0.95, 0.7),
               object_radius_rad=0.012,
               output_path='bh_video.mp4',
               figsize=(7, 7),
               title=None,
               draw_theoretical_circle=None,
               verbose=True):
    """
    Generate a video of an object following `trajectory` in the sky
    behind a black hole.

    Parameters
    ----------
    skymap : dict
        Output of get_or_compute_skymap.
    background : ndarray or callable
        Background sky; constant over the entire video.
    trajectory : callable
        Function t ∈ [0, 1] → (theta, phi), object position at time t.
    n_frames : int
        Number of frames to render.
    fps : int
        Frames per second in the output video.
    object_color : tuple
        RGB color of the object, values in [0, 1].
    object_radius_rad : float
        Angular radius of the object (typical: 0.005 to 0.02 rad,
        i.e. 0.3° to 1.1°).
    output_path : str
        Output file. Extension .mp4 or .gif.
    figsize : tuple
        Figure size in inches.
    title : str, optional
        Title displayed on each frame.
    draw_theoretical_circle : float, optional
        If given, draws the theoretical shadow circle (angle in radians).
    verbose : bool
        Print progress.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)

    pos = trajectory(0.0)
    overlay = [{'theta': pos[0], 'phi': pos[1],
                'color': object_color, 'radius_rad': object_radius_rad}]
    frame0 = render(skymap, background, overlays=overlay)
    im = ax.imshow(frame0, extent=[-1, 1, -1, 1], origin='upper')

    if draw_theoretical_circle is not None:
        th = np.linspace(0, 2 * np.pi, 200)
        r = draw_theoretical_circle
        ax.plot(r * np.cos(th), r * np.sin(th),
                'r--', lw=1.0, alpha=0.7)

    def update(k):
        t = k / max(n_frames - 1, 1)
        pos = trajectory(t)
        overlay = [{'theta': pos[0], 'phi': pos[1],
                    'color': object_color, 'radius_rad': object_radius_rad}]
        frame = render(skymap, background, overlays=overlay)
        im.set_array(frame)
        if verbose and k % max(1, n_frames // 10) == 0:
            print(f"  frame {k}/{n_frames}")
        return [im]

    anim = FuncAnimation(fig, update, frames=n_frames, blit=True)

    ext = Path(output_path).suffix.lower()
    if ext == '.gif':
        writer = PillowWriter(fps=fps)
    elif ext in ('.mp4', '.mov', '.mkv'):
        writer = FFMpegWriter(fps=fps, bitrate=4000)
    else:
        raise ValueError(f"Extension non supportée : {ext} "
                         "(utiliser .mp4, .mov, .mkv ou .gif)")

    if verbose:
        print(f"Writing {output_path}…")
    anim.save(output_path, writer=writer)
    plt.close(fig)
    if verbose:
        print(f"  done.")
