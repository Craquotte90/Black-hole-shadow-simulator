# Black-hole-shadow-simulator

Backward ray-tracing of null geodesics in the Schwarzschild metric, used to
generate still images and videos of a black-hole shadow distorting a stellar
background. Includes a comparison with flat spacetime to highlight
the gravitational lensing effect (Einstein ring formation).

This repository was developed as part of a second-year research project at
**École Nationale des Ponts et Chaussées**, IMI Department (2025–2026).

---

## English

### Purpose

This code simulates what a static observer would see when looking at a
Schwarzschild black hole in front of a stellar background, by integrating
null geodesics from each camera pixel back to the celestial sphere
(*backward ray-tracing*). It produces:

- **Still images** of the black-hole shadow and lensed background.
- **Videos** of a luminous object passing behind the black hole, showing the
  formation of an Einstein ring.
- **Comparison sequences** between Schwarzschild and flat spacetime, with
  identical camera, trajectory and background, to isolate the effect of the
  metric.

A set of videos (`einstein_ring_bh.mp4`, `einstein_ring_flat.mp4`) is provided as an example.
The integration is validated against the analytical shadow size
`sin α_shadow = (3√3 M / r_obs) √(1 - 2M/r_obs)`, with relative agreement
better than 1 % in our reference configurations.

### Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Craquotte90/Black-hole-shadow-simulator.git
   cd Black-hole-shadow-simulator
   ```

2. **Install Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Install ffmpeg for MP4 output.**

4. **Provide a background sky image.**

   Place an equirectangular panorama (width ≈ 2 × height, e.g. a Milky Way
   image) in the execution folder, or use NASA's panorama of the Milky Way, already provided (`milkyway.jpg`).

   Recommended free sources:
   - ESO Milky Way panorama (Brunier): <https://www.eso.org/public/images/eso0932a/>
   - ESO Milky Way panorama (Beletsky): <https://www.eso.org/public/images/eso1242a/>

### Running

Generate the comparison videos:

```bash
python main.py
```

By default this produces two MP4 files (Schwarzschild vs flat spacetime) for
each configured trajectory, using `r_obs = 50 M`, FOV = 30°, N = 400.

On the **first run**, the script computes the pixel → sky map and stores it
under `cache/skymap_robs50M_theta90deg_N400_fov30deg.npz`. This step can take some time. 
On subsequent runs the cache is reloaded.

> See `einstein_ring_bh.mp4` for an
> example of the expected output.

### File structure

| File | Role |
|---|---|
| `geodesics.py` | Schwarzschild metric, Christoffel symbols, RHS of the geodesic equation, conserved quantities. |
| `integrator.py` | Wrapper around `scipy.integrate.solve_ivp` (DOP853) with built-in conservation diagnostics and event factories. |
| `shadow.py` | Tetrad construction, pixel → ray → 4-velocity pipeline, single-pass image renderer. |
| `skymap_cache.py` | Compute and cache the pixel → sky map per camera configuration. |
| `bh_render.py` | Static image rendering from a precomputed skymap. |
| `bh_video.py` | Video generation with moving foreground overlays. |
| `main.py` | Main entry script: build skymaps, generate videos. |

---

## Français

### Objectif

Ce code simule ce que verrait un observateur statique en regardant un trou
noir de Schwarzschild devant un fond stellaire, en intégrant les géodésiques
nulles depuis chaque pixel de la caméra jusqu'à la sphère céleste
(*backward ray-tracing*). Il produit :

- **Images statiques** de l'ombre du trou noir et du fond déformé par
  lentille gravitationnelle.
- **Vidéos** d'un objet lumineux passant derrière le trou noir, montrant
  la formation d'un anneau d'Einstein.
- **Séquences comparatives** entre Schwarzschild et espace plat, avec
  caméra, trajectoire et fond identiques, pour isoler l'effet de la
  métrique.

Deux vidéos (`einstein_ring_bh.mp4`, `einstein_ring_flat.mp4`) sont fournies
en guise d'exemple.
L'intégration est validée contre la taille angulaire analytique de l'ombre
`sin α_shadow = (3√3 M / r_obs) √(1 - 2M/r_obs)`, avec un accord relatif
meilleur que 1 % dans les configurations de référence.

### Installation

1. **Cloner le dépôt :**

   ```bash
   git clone https://github.com/Craquotte90/Black-hole-shadow-simulator.git
   cd Black-hole-shadow-simulator
   ```

2. **Installer les dépendances Python :**

   ```bash
   pip install -r requirements.txt
   ```

3. **(Optionnel) Installer ffmpeg pour les sorties MP4.**

4. **Fournir une image de fond stellaire.**

   Placez un panorama équirectangulaire (largeur ≈ 2 × hauteur, par exemple
   une image de la Voie Lactée) dans le dossier courant de l'exécution, ou
   utiliser celle déjà fournie (`milkyway.jpg`).

   Sources libres recommandées :
   - Panorama Voie Lactée ESO (Brunier) : <https://www.eso.org/public/images/eso0932a/>
   - Panorama Voie Lactée ESO (Beletsky) : <https://www.eso.org/public/images/eso1242a/>

### Exécution

Générer les vidéos comparatives :

```bash
python main.py
```

Par défaut, le script produit deux fichiers MP4 (Schwarzschild vs espace plat)
pour chaque trajectoire configurée, avec `r_obs = 50 M`, FOV = 30°, N = 400.

Lors du **premier lancement**, le script calcule la carte pixel → ciel et la
sauve sous `cache/skymap_robs50M_theta90deg_N400_fov30deg.npz`. Cette étape
peut prendre du temps. Les exécutions suivantes rechargent le cache.

> Voir `einstein_ring_bh.mp4` pour un exemple 
> de la sortie attendue.

### Structure des fichiers

| Fichier | Rôle |
|---|---|
| `geodesics.py` | Métrique de Schwarzschild, symboles de Christoffel, second membre de l'équation géodésique, quantités conservées. |
| `integrator.py` | Wrapper autour de `scipy.integrate.solve_ivp` (DOP853) avec diagnostics de conservation et fabriques d'évènements. |
| `shadow.py` | Construction de la tétrade, pipeline pixel → rayon → 4-vitesse, rendu d'image. |
| `skymap_cache.py` | Calcul et mise en cache de la carte pixel → ciel pour chaque configuration caméra. |
| `bh_render.py` | Rendu d'images statiques à partir d'un skymap pré-calculé. |
| `bh_video.py` | Génération de vidéos avec objets mobiles superposés. |
| `main.py` | Script principal : construction des skymaps, génération des vidéos. |


---

## Authors

Ayush Adhikari, Basile Boeuf, Alexandre Décavé, Pierre Gineste,
Alizée Montigny, Ilyas Zeriouel
— École Nationale des Ponts et Chaussées (IMI Department), 2025–2026.

Supervisor: Vincent Boulard.
