# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from .ssl_meta_arch import SSLMetaArch


def get_args_parser(*args, **kwargs):
    raise RuntimeError(
        "The upstream DINOv2 train.py entrypoint has been retired in this codebase. "
        "Use the PyTorch Lightning wrapper in methods/dinov2.py instead."
    )


def main(*args, **kwargs):
    raise RuntimeError(
        "The upstream DINOv2 train.py entrypoint has been retired in this codebase. "
        "Use the PyTorch Lightning wrapper in methods/dinov2.py instead."
    )
