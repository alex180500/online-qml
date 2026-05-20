from typing import Iterable

_SHADOW_METHOD_FLAGS: dict[str, tuple[bool, bool]] = {
    "ost": (False, False),
    "state_prior_ost": (True, False),
    "povm_prior_ost": (False, True),
    "prior_ost": (True, True),
}
_LINEAR_METHODS = frozenset({"pinv", "ridge"})
_ALL_METHODS = frozenset(_SHADOW_METHOD_FLAGS) | _LINEAR_METHODS


def shadow_method_flags(method: str) -> tuple[bool, bool]:
    """Return state-prior and POVM-prior flags for a shadow method."""
    try:
        return _SHADOW_METHOD_FLAGS[method]
    except KeyError as exc:
        raise ValueError(f"Unknown shadow method '{method}'.") from exc


def split_methods(methods: Iterable[str]) -> tuple[list[str], list[str]]:
    """Split method names into shadow and linear methods."""
    methods = list(methods)
    unknown = sorted(set(methods) - _ALL_METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    shadow_methods = [method for method in methods if method in _SHADOW_METHOD_FLAGS]
    linear_methods = [method for method in methods if method in _LINEAR_METHODS]
    return shadow_methods, linear_methods
