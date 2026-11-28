"""Settings shared by the tests: the device they run on and the tier.

Two environment variables, read once:

    WARP_FEBIO_TEST_DEVICE   "cpu" (the default), "cuda:0", "cuda:1", ...
    WARP_FEBIO_TEST_LEVEL    "quick" (the default) or "full"

A test module that uses Warp imports DEVICE, setUpModule and tearDownModule
from here. While the module's tests run, DEVICE is Warp's current device, so
calls that name no device follow it; calls that name one pass device=DEVICE.
A test that compares CUDA with CPU on purpose names both devices.

The quick tier is one representative case per area and finishes in a few
minutes on CPU. The full tier adds every FEBio suite parameter set, every
linear solver on every case and the larger problems. Tests in the full tier
carry the full_tier decorator, so in a quick run they show in the skip count
with the reason, as CONTRIBUTING asks.

This file is not a test module: unittest discovery looks for test*.py.
"""

import os
import unittest

DEVICE = os.environ.get("WARP_FEBIO_TEST_DEVICE", "cpu")

LEVEL = os.environ.get("WARP_FEBIO_TEST_LEVEL", "quick")
if LEVEL not in ("quick", "full"):
    raise ValueError(f'WARP_FEBIO_TEST_LEVEL must be "quick" or "full", not {LEVEL!r}')
FULL = LEVEL == "full"


def full_tier(reason: str):
    """Run this test or class only in the full tier. The reason says what it adds."""
    return unittest.skipUnless(FULL, f"full tier ({reason}); set WARP_FEBIO_TEST_LEVEL=full")


_scopes = []


def setUpModule():
    """Make DEVICE Warp's current device for the importing module, if Warp is installed."""
    try:
        import warp as wp
    except ImportError:  # the model and file layers are tested without Warp too
        return
    _scopes.append(wp.ScopedDevice(DEVICE))
    _scopes[-1].__enter__()


def tearDownModule():
    while _scopes:
        _scopes.pop().__exit__(None, None, None)
