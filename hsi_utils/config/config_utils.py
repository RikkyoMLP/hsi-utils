from omegaconf import OmegaConf
from pathlib import Path
from typing import List, Union, Optional
from hsi_utils.templates import get_template
import sys


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
    cli_config = OmegaConf.from_cli(args)

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


def from_args(base_config_path: Union[Path, str, None] = None, inject_template: bool = False) -> OmegaConf:
    args = sys.argv[1:]
    return load_merge_config(args, base_config_path, inject_template)