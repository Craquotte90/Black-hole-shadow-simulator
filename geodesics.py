import numpy as np

def metric(r, theta, M=1.0):
    """
    Composantes covariantes g_{μν}(r, θ) de la métrique de Schwarzschild.
    Signature (-, +, +, +). Unités normalisées (G = c = 1).
    Retourne une matrice 4x4.
    """
    f = 1.0 - 2.0 * M / r
    sin_t = np.sin(theta)
    return np.array([
        [-f, 0.0, 0.0, 0.0],
        [0.0, 1.0/f, 0.0, 0.0],
        [0.0, 0.0, r**2, 0.0],
        [0.0, 0.0, 0.0, r**2 * sin_t**2],
    ])


def inverse_metric(r, theta, M=1.0):
    """
    Composantes contravariantes g^{μν}, simplement les inverses des termes
    diagonaux puisque g est diagonale en coordonnées de Schwarzschild.
    """
    f = 1.0 - 2.0 * M / r
    sin_t = np.sin(theta)
    return np.array([
        [-1.0/f, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, 1.0 / r**2, 0.0],
        [0.0, 0.0, 0.0, 1.0 / (r**2 * sin_t**2)],
    ])


def christoffels(r, theta, M=1.0):
    """
    Symboles de Christoffel Γ^α_{βγ} de la connexion de Levi-Civita
    dans la métrique de Schwarzschild.

    Convention: Gamma[a, b, c] = Γ^a_{bc}.
    Symétrie en (b, c): Gamma[a, b, c] = Gamma[a, c, b].
    Retourne un tableau 4x4x4.
    """
    Gamma = np.zeros((4, 4, 4))
    f = 1.0 - 2.0 * M / r
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    # Γ^t_{tr} = Γ^t_{rt}
    Gamma[0, 0, 1] = Gamma[0, 1, 0] = M / (r**2 * f)

    # Γ^r_{tt}
    Gamma[1, 0, 0] = M * f / r**2
    # Γ^r_{rr}
    Gamma[1, 1, 1] = -M / (r**2 * f)
    # Γ^r_{θθ}
    Gamma[1, 2, 2] = -(r - 2.0 * M)
    # Γ^r_{φφ}
    Gamma[1, 3, 3] = -(r - 2.0 * M) * sin_t**2

    # Γ^θ_{rθ} = Γ^θ_{θr}
    Gamma[2, 1, 2] = Gamma[2, 2, 1] = 1.0 / r
    # Γ^θ_{φφ}
    Gamma[2, 3, 3] = -sin_t * cos_t

    # Γ^φ_{rφ} = Γ^φ_{φr}
    Gamma[3, 1, 3] = Gamma[3, 3, 1] = 1.0 / r
    # Γ^φ_{θφ} = Γ^φ_{φθ}
    Gamma[3, 2, 3] = Gamma[3, 3, 2] = cos_t / sin_t

    return Gamma


def V_eps(r, L, eps, M=1.0):
    """
    Potentiel effectif radial V_ε(r) = (1 - 2M/r)(L²/r² - ε).
    ε = -1 → timelike (Mercure).
    ε =  0 → null (photon).
    """
    f = 1.0 - 2.0 * M / r
    return f * (L**2 / r**2 - eps)


def V_eps_prime(r, L, eps, M=1.0):
    """
    Dérivée V'_ε(r) en r. Utile pour localiser les orbites circulaires
    (V'=0) et tester analytiquement la stabilité (signe de V'').
    """
    return (2.0 * M / r**2) * (L**2 / r**2 - eps) + (1.0 - 2.0 * M / r) * (-2.0 * L**2 / r**3)


def geodesic_rhs(lam, y, M=1.0):
    """
    Membre de droite de l'équation géodésique vue comme un système
    différentiel du premier ordre dans y ∈ R^8.

    Signature compatible avec scipy.integrate.solve_ivp:
    f(lam, y) → dy/dλ.

    Paramètres
    ----------
    lam : float (paramètre affine, non utilisé car l'équation est
                 autonome en λ; argument requis par solve_ivp)
    y   : array (8,) [t, r, θ, φ, ṫ, ṙ, θ̇, φ̇]
    M   : masse du corps central

    Retour
    ------
    dydlam : array (8,) [ṫ, ṙ, θ̇, φ̇, ẗ, r̈, θ̈, φ̈]
    """
    r, theta = y[1], y[2]
    v = y[4:]
    Gamma = christoffels(r, theta, M)
    a = -np.einsum('abc,b,c->a', Gamma, v, v)
    return np.concatenate([v, a])


def conserved_quantities(y, M=1.0):
    """
    Calcule à partir de l'état y = (x, ẋ):
        E    = -g(γ̇, ∂_t) = (1 - 2M/r) ṫ
        L    =  g(γ̇, ∂_φ) = r² sin²θ φ̇
        norm =  g(γ̇, γ̇)
    Ces trois quantités sont conservées exactement le long
    de toute géodésique. Tout écart numérique mesure l'erreur
    d'intégration.
    """
    r, theta = y[1], y[2]
    tdot, rdot, thdot, phdot = y[4:]
    f = 1.0 - 2.0 * M / r
    sin_t = np.sin(theta)

    E = f * tdot
    L = r**2 * sin_t**2 * phdot
    norm = (-f * tdot**2
            + rdot**2 / f
            + r**2 * thdot**2
            + r**2 * sin_t**2 * phdot**2)
    return E, L, norm


def conservation_errors(y_history, M=1.0):
    """
    À partir d'un historique d'états y_history (shape (N, 8)),
    retourne les variations relatives de E, L et |norm - norm_0|
    le long de la trajectoire.

    Usage typique : tracer ces trois courbes en fonction de λ
    et vérifier qu'elles restent < 1e-10.
    """
    E0, L0, n0 = conserved_quantities(y_history[0], M)
    deltas = np.array([
        [(E - E0) / max(abs(E0), 1e-30),
         (L - L0) / max(abs(L0), 1e-30),
         (n - n0)]
        for E, L, n in (conserved_quantities(y, M) for y in y_history)
    ])
    return deltas
