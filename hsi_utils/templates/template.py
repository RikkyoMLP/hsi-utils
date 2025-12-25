from typing import Tuple, Optional, List, Dict, Union, Any
from omegaconf import DictConfig

configs: List[Dict[str, Optional[str]]] = [
        {"name_includes": "mst", "input_setting": "H", "input_mask": "Phi"},
        {"name_includes": "gap_net", "input_setting": "Y", "input_mask": "Phi_PhiPhiT"},
        {
            "name_includes": "admm_net",
            "input_setting": "Y",
            "input_mask": "Phi_PhiPhiT",
        },
        {"name_includes": "dnu", "input_setting": "Y", "input_mask": "Phi_PhiPhiT"},
        {"name_includes": "dauhst", "input_setting": "Y", "input_mask": "Phi_PhiPhiT"},
        {"name_includes": "tsa_net", "input_setting": "HM", "input_mask": None},
        {"name_includes": "hdnet", "input_setting": "H", "input_mask": None},
        {"name_includes": "dgsmp", "input_setting": "Y", "input_mask": None},
        {"name_includes": "birnat", "input_setting": "Y", "input_mask": "Phi"},
        {"name_includes": "mst_plus_plus", "input_setting": "H", "input_mask": "Mask"},
        {"name_includes": "mst++", "input_setting": "H", "input_mask": "Mask"},
        {"name_includes": "bisrnet", "input_setting": "H", "input_mask": "Mask"},
        {"name_includes": "cst", "input_setting": "H", "input_mask": "Mask"},
        {"name_includes": "lambda_net", "input_setting": "Y", "input_mask": "Phi"},
    ]

def get_template(
    args: Union[DictConfig, Dict[str, Any]]
) -> Tuple[Optional[str], Optional[str]]:
    """
    Determine input_setting and input_mask based on args.template.
    Prioritizes user's explicit input if template is not specified.
    If template is specified, it overrides input_setting and input_mask.
    """
    # Handle both object-like access (OmegaConf/Namespace) and dict-like access
    if hasattr(args, "get"):
        template = args.get("template", "")
        input_setting = args.get("input_setting", None)
        input_mask = args.get("input_mask", None)
    else:
        # Fallback for simple Namespace objects
        template = getattr(args, "template", "")
        input_setting = getattr(args, "input_setting", None)
        input_mask = getattr(args, "input_mask", None)

    # If no template is specified, respect the original input_setting/mask
    if not template:
        return input_setting, input_mask

    # If template is specified, override settings
    for config in configs:
        if config["name_includes"] in template.lower():
            input_setting = config["input_setting"]
            input_mask = config["input_mask"]

    return input_setting, input_mask


def get_template_list() -> List[Dict[str, Optional[str]]]:
    """Get list of templates.

    Returns:
        `List[Dict[str, Optional[str]]]: List of templates.
    """
    return configs