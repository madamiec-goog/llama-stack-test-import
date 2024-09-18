# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse

from llama_stack.cli.subcommand import Subcommand
from llama_stack.distribution.datatypes import *  # noqa: F403
from functools import lru_cache
from pathlib import Path

import yaml


@lru_cache()
def available_templates_specs() -> List[BuildConfig]:
    template_specs = []
    for p in Path("llama_stack/distribution/templates/").rglob("*.yaml"):
        with open(p, "r") as f:
            build_config = BuildConfig(**yaml.safe_load(f))
            template_specs.append(build_config)

    return template_specs


class StackBuild(Subcommand):
    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "build",
            prog="llama stack build",
            description="Build a Llama stack container",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_stack_build_command)

    def _add_arguments(self):
        self.parser.add_argument(
            "--config",
            type=str,
            default=None,
            help="Path to a config file to use for the build. You can find example configs in llama_stack/distribution/example_configs. If this argument is not provided, you will be prompted to enter information interactively",
        )

        self.parser.add_argument(
            "--template",
            type=str,
            default=None,
            help="Name of the example template config to use for build. You may use `llama stack build --list-templates` to check out the available templates",
        )

        self.parser.add_argument(
            "--list-templates",
            type=bool,
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Show the available templates for building a Llama Stack distribution",
        )

        self.parser.add_argument(
            "--name",
            type=str,
            help="Name of the Llama Stack build to override from template config. This name will be used as paths to store configuration files, build conda environments/docker images. If not specified, will use the name from the template config. ",
        )

        # 1. llama stack build

        # 2. llama stack build --list-templates

        # 2. llama stack build --template <template_name> --name <name>

        # 3. llama stack build --config <config_path>

    def _run_stack_build_command_from_build_config(
        self, build_config: BuildConfig
    ) -> None:
        import json
        import os

        from llama_stack.distribution.build import ApiInput, build_image, ImageType

        from llama_stack.distribution.utils.config_dirs import DISTRIBS_BASE_DIR
        from llama_stack.distribution.utils.serialize import EnumEncoder
        from termcolor import cprint

        # save build.yaml spec for building same distribution again
        if build_config.image_type == ImageType.docker.value:
            # docker needs build file to be in the llama-stack repo dir to be able to copy over to the image
            llama_stack_path = Path(os.path.relpath(__file__)).parent.parent.parent
            build_dir = (
                llama_stack_path / "configs/distributions" / build_config.image_type
            )
        else:
            build_dir = (
                Path(os.getenv("CONDA_PREFIX")).parent
                / f"llamastack-{build_config.name}"
            )

        os.makedirs(build_dir, exist_ok=True)
        build_file_path = build_dir / f"{build_config.name}-build.yaml"

        with open(build_file_path, "w") as f:
            to_write = json.loads(json.dumps(build_config.dict(), cls=EnumEncoder))
            f.write(yaml.dump(to_write, sort_keys=False))

        build_image(build_config, build_file_path)

        cprint(
            f"Build spec configuration saved at {str(build_file_path)}",
            color="blue",
        )
        cprint(
            f"You may now run `llama stack configure {build_config.name}` or `llama stack configure {str(build_file_path)}`",
            color="green",
        )

    def _run_template_list_cmd(self, args: argparse.Namespace) -> None:
        import json

        from llama_stack.cli.table import print_table

        # eventually, this should query a registry at llama.meta.com/llamastack/distributions
        headers = [
            "Distribution Type",
            "Providers",
            "Image Type",
            "Description",
        ]

        rows = []
        for spec in available_templates_specs():
            rows.append(
                [
                    spec.name,
                    json.dumps(spec.distribution_spec.providers, indent=2),
                    spec.image_type,
                    spec.distribution_spec.description,
                ]
            )
        print_table(
            rows,
            headers,
            separate_rows=True,
        )

    def _run_stack_build_command(self, args: argparse.Namespace) -> None:
        from llama_stack.distribution.distribution import Api, api_providers
        from llama_stack.distribution.utils.dynamic import instantiate_class_type
        from prompt_toolkit import prompt
        from prompt_toolkit.validation import Validator
        from termcolor import cprint

        if args.list_templates:
            self._run_template_list_cmd(args)
            return

        if not args.config:
            name = prompt(
                "> Enter a name for your Llama Stack, this name will be used as paths to store configuration files, build conda environments and docker images (e.g. my-local-stack): "
            )
            image_type = prompt(
                "> Enter the image type you want your Llama Stack to be built as (docker or conda): ",
                validator=Validator.from_callable(
                    lambda x: x in ["docker", "conda"],
                    error_message="Invalid image type, please enter conda or docker",
                ),
                default="conda",
            )

            cprint(
                f"\n Llama Stack is composed of several APIs working together. Let's configure the providers (implementations) you want to use for these APIs.",
                color="green",
            )

            providers = dict()
            for api in Api:
                all_providers = api_providers()
                providers_for_api = all_providers[api]

                api_provider = prompt(
                    "> Enter the API provider for the {} API: (default=meta-reference): ".format(
                        api.value
                    ),
                    validator=Validator.from_callable(
                        lambda x: x in providers_for_api,
                        error_message="Invalid provider, please enter one of the following: {}".format(
                            providers_for_api.keys()
                        ),
                    ),
                    default=(
                        "meta-reference"
                        if "meta-reference" in providers_for_api
                        else list(providers_for_api.keys())[0]
                    ),
                )

                providers[api.value] = api_provider

            description = prompt(
                "\n > (Optional) Enter a short description for your Llama Stack: ",
                default="",
            )

            distribution_spec = DistributionSpec(
                providers=providers,
                description=description,
            )

            build_config = BuildConfig(
                name=name, image_type=image_type, distribution_spec=distribution_spec
            )
            self._run_stack_build_command_from_build_config(build_config)
            return

        with open(args.config, "r") as f:
            try:
                build_config = BuildConfig(**yaml.safe_load(f))
            except Exception as e:
                self.parser.error(f"Could not parse config file {args.config}: {e}")
                return
            if args.name:
                build_config.name = args.name
            self._run_stack_build_command_from_build_config(build_config)
