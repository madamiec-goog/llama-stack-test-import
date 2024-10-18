# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from llama_stack.apis.datasets import *  # noqa: F403
from ..registry import Registry


DatasetRegistry = Registry[BaseDataset]()
