"""
experiments/configs.py
══════════════════════
Experiment configuration variants for systematic comparison.

Each experiment is a dict of config overrides that gets deep-merged
onto the base config.yaml. Only specify the keys that differ.

Experiments are immutable channel ablations. Every arm explicitly states which
predictive channels are enabled, so changes to the base config cannot turn two
named experiments into accidental duplicates.
"""

EXPERIMENTS = {
    "market_only": {
        "description": "SAC with causal market/portfolio features only",
        "overrides": {
            "alpha": {"enabled": False},
            "patchtst": {"enabled": False},
            "gnn": {"enabled": False},
            "vae_regime": {"enabled": False},
            "meta_model": {"enabled": False},
        },
    },

    "lgbm_only": {
        "description": "SAC plus purged-CV LightGBM alpha",
        "overrides": {
            "alpha": {"enabled": True},
            "patchtst": {"enabled": False},
            "gnn": {"enabled": False},
            "vae_regime": {"enabled": False},
            "meta_model": {"enabled": False},
        },
    },

    "lgbm_patchtst": {
        "description": "SAC plus Ridge-combined LightGBM and PatchTST",
        "overrides": {
            "alpha": {"enabled": True},
            "patchtst": {"enabled": True},
            "gnn": {"enabled": False},
            "vae_regime": {"enabled": False},
            "meta_model": {"enabled": True},
        },
    },

    "lgbm_gnn": {
        "description": "SAC plus Ridge-combined LightGBM and corrected GNN",
        "overrides": {
            "alpha": {"enabled": True},
            "patchtst": {"enabled": False},
            "gnn": {"enabled": True},
            "vae_regime": {"enabled": False},
            "meta_model": {"enabled": True},
        },
    },

    "lgbm_vae": {
        "description": "SAC plus LightGBM alpha and separate VAE regime state",
        "overrides": {
            "alpha": {"enabled": True},
            "patchtst": {"enabled": False},
            "gnn": {"enabled": False},
            "vae_regime": {"enabled": True},
            "meta_model": {"enabled": False},
        },
    },

    "full_corrected": {
        "description": "Corrected LightGBM, PatchTST, GNN, VAE and Ridge stack",
        "overrides": {
            "alpha": {"enabled": True},
            "patchtst": {"enabled": True},
            "gnn": {"enabled": True},
            "vae_regime": {"enabled": True},
            "meta_model": {"enabled": True},
        },
    },
}


def get_experiment_names():
    return list(EXPERIMENTS.keys())


def get_experiment(name: str) -> dict:
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {name}. Choose from: {list(EXPERIMENTS.keys())}")
    return EXPERIMENTS[name]
