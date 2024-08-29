# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import json
from pathlib import Path

import yaml
from termcolor import cprint

from llama_toolchain.cli.subcommand import Subcommand
from llama_toolchain.common.config_dirs import BUILDS_BASE_DIR
from llama_toolchain.distribution.datatypes import *  # noqa: F403


class StackConfigure(Subcommand):
    """Llama cli for configuring llama toolchain configs"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "configure",
            prog="llama stack configure",
            description="configure a llama stack distribution",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_stack_configure_cmd)

    def _add_arguments(self):
        from llama_toolchain.distribution.package import BuildType
        from llama_toolchain.distribution.registry import available_distribution_specs

        self.parser.add_argument(
            "--build-name",
            type=str,
            help="(Fully qualified) name of the stack build to configure. Alternatively, provider --distribution and --name",
            required=False,
        )
        allowed_ids = [d.distribution_id for d in available_distribution_specs()]
        self.parser.add_argument(
            "--distribution",
            type=str,
            choices=allowed_ids,
            help="Distribution (one of: {})".format(allowed_ids),
            required=False,
        )
        self.parser.add_argument(
            "--name",
            type=str,
            help="Name of the build",
            required=False,
        )
        self.parser.add_argument(
            "--type",
            type=str,
            default="conda_env",
            choices=[v.value for v in BuildType],
        )

    def _run_stack_configure_cmd(self, args: argparse.Namespace) -> None:
        from llama_toolchain.distribution.package import BuildType

        if args.build_name:
            name = args.build_name
            if name.endswith(".yaml"):
                name = name.replace(".yaml", "")
        else:
            build_type = BuildType(args.type)
            name = f"{build_type.descriptor()}-{args.distribution}-{args.name}"

        config_file = BUILDS_BASE_DIR / "stack" / f"{name}.yaml"
        if not config_file.exists():
            self.parser.error(
                f"Could not find {config_file}. Please run `llama stack build` first"
            )
            return

        configure_llama_distribution(config_file)


def configure_llama_distribution(config_file: Path) -> None:
    from llama_toolchain.common.serialize import EnumEncoder
    from llama_toolchain.distribution.configure import configure_api_providers
    from llama_toolchain.distribution.registry import resolve_distribution_spec

    with open(config_file, "r") as f:
        config = PackageConfig(**yaml.safe_load(f))

    dist = resolve_distribution_spec(config.distribution_id)
    if dist is None:
        raise ValueError(
            f"Could not find any registered distribution `{config.distribution_id}`"
        )

    if config.providers:
        cprint(
            f"Configuration already exists for {config.distribution_id}. Will overwrite...",
            "yellow",
            attrs=["bold"],
        )

    config.providers = configure_api_providers(config.providers)

    with open(config_file, "w") as fp:
        to_write = json.loads(json.dumps(config.dict(), cls=EnumEncoder))
        fp.write(yaml.dump(to_write, sort_keys=False))

    print(f"YAML configuration has been written to {config_file}")
