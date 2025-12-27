from omegaconf import OmegaConf
from pathlib import Path
from typing import List, Union, Optional
from hsi_utils.templates import get_template
import argparse


def load_merge_config(
    args: List[str],
    base_config_path: Union[Path, str, None] = None,
    inject_template: bool = False,
) -> OmegaConf:
    """Load base config and merge with CLI arguments.

    Args:
        args (List[str]): CLI arguments
        base_config_path (Union[Path, str, None], optional): Base config path. Defaults to None.
        inject_template (bool, optional): Whether template should be injected. Defaults to False.

    Returns:
        OmegaConf: Merged config
    """
    base_config = (
        OmegaConf.create()
        if base_config_path is None
        else OmegaConf.load(base_config_path)
    )
    if len(args) > 0 and isinstance(args, dict):
        args = OmegaConf.create(args)
    else:
        args = OmegaConf.from_cli(args)

    cli_config = args

    # First merge to get the potential 'template' value from CLI overriding base
    merged_config = OmegaConf.merge(base_config, cli_config)

    # If template is explicitly specified, force inject
    if hasattr(merged_config, "template") and merged_config.template is not None:
        inject_template = True

    if inject_template:
        # Resolve template dependencies
        input_setting, input_mask = get_template(merged_config)

        updates = {}
        if input_setting is not None:
            updates["input_setting"] = input_setting
        if input_mask is not None:
            updates["input_mask"] = input_mask

        if updates:
            merged_config = OmegaConf.merge(merged_config, OmegaConf.create(updates))

    return merged_config


def from_args(
    base_config_path: Union[Path, str, None] = None, inject_template: bool = False
) -> OmegaConf:
    # args = sys.argv[1:]
    args = {}

    def _transform_key(key: str) -> str:
        if key.startswith("--"):
            return key[2:]
        return key
    
    def _transform_value(value: str) -> Union[str, bool, int, float]:
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value

    _, unknown = argparse.ArgumentParser().parse_known_args()
    for i in range(0, len(unknown), 2):
        if i + 1 >= len(unknown):
            break
        key = unknown[i]
        value = unknown[i + 1]
        args[_transform_key(key)] = _transform_value(value)

    return load_merge_config(args, base_config_path, inject_template)
