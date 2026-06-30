# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
from typing import Any


RETIRED_MESSAGE = (
    "The upstream architectures.dinov2.train.train entrypoint has been retired in this codebase. "
    "Use the PyTorch Lightning wrapper in methods/dinov2.py instead."
)


def _raise_retired() -> None:
    raise RuntimeError(RETIRED_MESSAGE)


def get_args_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Retired DINOv2 training entrypoint",
        add_help=add_help,
        description=RETIRED_MESSAGE,
    )
    return parser


def build_optimizer(*args: Any, **kwargs: Any) -> Any:
    _raise_retired()


def build_schedulers(*args: Any, **kwargs: Any) -> Any:
    _raise_retired()


def apply_optim_scheduler(*args: Any, **kwargs: Any) -> Any:
    _raise_retired()


def do_test(*args: Any, **kwargs: Any) -> Any:
    _raise_retired()


def do_train(*args: Any, **kwargs: Any) -> Any:
    _raise_retired()


def main(*args: Any, **kwargs: Any) -> Any:
    _raise_retired()


if __name__ == "__main__":
    main()
