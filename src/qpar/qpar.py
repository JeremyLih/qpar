"""Q-Par (protocol Section 8): definition, the legacy v2 form, sensitivity settings."""

EXPONENTS = ("alpha", "beta", "gamma", "delta", "epsilon")
EQUAL = dict.fromkeys(EXPONENTS, 1.0)


def qpar(A, sigma_a, Q, D, G, Q_max, D_max, G_max, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0, epsilon=1.0):
    """Q-Par = A^α (1−σ_a)^β / [(Q/Q_max)^γ (D/D_max)^δ (G/G_max)^ε]; elementwise on arrays."""
    return (A ** alpha * (1 - sigma_a) ** beta
            / ((Q / Q_max) ** gamma * (D / D_max) ** delta * (G / G_max) ** epsilon))


def qpar_legacy(A, sigma_a, Q, D, G, D_max, G_max, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0, epsilon=1.0):
    """The v2 manuscript's form, raw Q in the denominator. Only used to check arithmetic (8.1)."""
    return (A ** alpha * (1 - sigma_a) ** beta
            / (Q ** gamma * (D / D_max) ** delta * (G / G_max) ** epsilon))


def sensitivity_settings():
    """8.2: equal weights, each exponent at 0.5 and 2 (others 1), and the three presets."""
    settings = {"equal": EQUAL}
    for name in EXPONENTS:
        for value in (0.5, 2.0):
            settings[f"{name}={value:g}"] = {**EQUAL, name: value}
    settings["accuracy-first"] = {**EQUAL, "alpha": 2.0}
    settings["latency-first"] = {**EQUAL, "delta": 2.0}
    settings["resource-first"] = {**EQUAL, "gamma": 2.0, "epsilon": 2.0}
    return settings
