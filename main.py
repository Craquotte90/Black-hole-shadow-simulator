from skymap_cache import get_or_compute_skymap, flat_skymap
from bh_render import load_texture
from bh_video import make_video, linear_sky_trajectory
import numpy as np


def compute_skymaps(N, fov, r_obs):
    # Deux skymaps : Schwarzschild (calculé/chargé) et espace plat (instantané)
    skymap_bh   = get_or_compute_skymap(N=N, fov=fov, r_obs=r_obs)
    skymap_flat = flat_skymap(N=N, fov=fov)
    return skymap_bh, skymap_flat

# Panorama de la Voie Lactée, peut être remplacé par n'importe quelle autre image de même taille
def load_background_texture(filename="milkyway.jpg"):
    return load_texture(filename, target_size=(2048, 1024))

#Quelques propositions de trajectoires pour l'animation de l'anneau d'Einstein



def traj_video(traj_list, skymap_bh, skymap_flat, bg):
    for i, traj in enumerate(traj_list):
        # Vidéo avec trou noir
        make_video(skymap_bh, bg, traj,
                n_frames=200, fps=25,
                object_color=(1.0, 0.95, 0.7), object_radius_rad=0.012,
                output_path=f"einstein_ring_bh_{i}.mp4",
                title="Avec trou noir de Schwarzschild")

        # Vidéo sans trou noir
        make_video(skymap_flat, bg, traj,
                n_frames=200, fps=25,
                object_color=(1.0, 0.95, 0.7), object_radius_rad=0.012,
                output_path=f"einstein_ring_flat_{i}.mp4",
                title="Sans trou noir (espace plat)")


if __name__ == "__main__":
    # Paramètres de la simulation, modiffiables
    N = 400
    fov = np.deg2rad(30)
    r_obs = 50.0
    texture_img = "milkyway.jpg"

    # trajectoires à tester, modifiables
    traj2 = linear_sky_trajectory(
        np.pi/2 - 0.10, np.pi - 0.35,
        np.pi/2 - 0.10, np.pi + 0.35,
    )

    traj3 = linear_sky_trajectory(
        np.pi/2, np.pi - 0.30,
        np.pi/2, np.pi + 0.30,
    )

    traj4 = linear_sky_trajectory(
        np.pi/2, np.pi - 0.10,
        np.pi/2, np.pi + 0.10,
    )


    traj_list = [traj3]
    skymap_bh, skymap_flat = compute_skymaps(N, fov, r_obs)
    bg = load_background_texture(texture_img)
    traj_video(traj_list, skymap_bh, skymap_flat, bg)




    