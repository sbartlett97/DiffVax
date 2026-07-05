"""Immunization strategies for DiffVax.

Submodules are imported lazily (PEP 562) so that optional heavy dependencies
of the baseline methods (e.g. ``cv2`` for DiffusionGuard) are only required
when the corresponding class is actually used — core DiffVax training must
not depend on them.
"""

__all__ = [
    "DiffVaxImmunization",
    "PhotoGuardImmunization",
    "PhotoGuardDiffusionImmunization",
    "DiffusionGuardImmunization",
]

_LAZY = {
    "DiffVaxImmunization": ".diffvax_immunization",
    "PhotoGuardImmunization": ".photoguard_immunization",
    "PhotoGuardDiffusionImmunization": ".photoguard_immunization",
    "DiffusionGuardImmunization": ".diffusionguard_immunization",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(_LAZY[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
