"""Bridge-style Franka pick-and-place experiment for Isaac Lab."""

try:
    from .bridge_action_mapper import BridgeActionMapper, BridgeActionMappingConfig

    __all__ = ["BridgeActionMapper", "BridgeActionMappingConfig"]
except ImportError:
    # When run as a standalone script directory (e.g. pytest from tests/),
    # relative imports are not available. This is expected.
    pass
