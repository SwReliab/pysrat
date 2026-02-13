from __future__ import annotations

from .models.exp import ExponentialNHPP

SRM_MODELS = {
    "exp": ExponentialNHPP,
}


def srm(names):
    """
    names: str or list[str]
    returns: model instance or list[model instance]
    """
    if isinstance(names, str):
        if names not in SRM_MODELS:
            raise ValueError(f"Unknown model name: {names}")
        return SRM_MODELS[names]()
    return [srm(n) for n in names]
