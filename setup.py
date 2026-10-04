"""Build script. Metadata lives in setup.cfg; this adds the native extension.

The extension (rust/) is optional. A build without a working Rust toolchain
still succeeds and produces a pure-Python install that parses with lxml.
Set FASTFEEDPARSER_PURE=1 to skip the extension on purpose, which is how the
py3-none-any fallback wheel is built.
"""

import os
import platform
import sys
import sysconfig

from setuptools import setup
from setuptools_rust import Binding, RustExtension

# The extension uses the stable ABI from CPython 3.9, which other
# interpreters and free-threaded builds do not provide.
_SUPPORTS_EXTENSION = (
    sys.version_info >= (3, 9)
    and platform.python_implementation() == "CPython"
    and not sysconfig.get_config_var("Py_GIL_DISABLED")
)
_PURE = bool(os.environ.get("FASTFEEDPARSER_PURE")) or not _SUPPORTS_EXTENSION

if _PURE:
    setup()
else:
    setup(
        rust_extensions=[
            RustExtension(
                "fastfeedparser._core",
                path="rust/Cargo.toml",
                binding=Binding.PyO3,
                features=["extension-module"],
                py_limited_api=True,
                optional=True,
                debug=False,
            )
        ],
        # One wheel per platform serves every CPython from 3.9 on.
        options={"bdist_wheel": {"py_limited_api": "cp39"}},
        zip_safe=False,
    )
