from typing import Any, Tuple, Optional, List, Dict


def get_template(args: Any) -> Tuple[Optional[str], Optional[str]]:
    """
    Determine input_setting and input_mask based on args.template.
    """
    configs: List[Dict[str, Optional[str]]] = [
        {"name_includes": "mst", "input_setting": "H", "input_mask": "Phi"},
        {"name_includes": "gap_net", "input_setting": "Y", "input_mask": "Phi_PhiPhiT"},
        {"name_includes": "admm_net", "input_setting": "Y", "input_mask": "Phi_PhiPhiT"},
        {"name_includes": "dnu", "input_setting": "Y", "input_mask": "Phi_PhiPhiT"},
        {"name_includes": "dauhst", "input_setting": "Y", "input_mask": "Phi_PhiPhiT"},
        {"name_includes": "tsa_net", "input_setting": "HM", "input_mask": None},
        {"name_includes": "hdnet", "input_setting": "H", "input_mask": None},
        {"name_includes": "dgsmp", "input_setting": "Y", "input_mask": None},
        {"name_includes": "birnat", "input_setting": "Y", "input_mask": "Phi"},
        {"name_includes": "mst_plus_plus", "input_setting": "H", "input_mask": "Mask"},
        {"name_includes": "bisrnet", "input_setting": "H", "input_mask": "Mask"},
        {"name_includes": "cst", "input_setting": "H", "input_mask": "Mask"},
        {"name_includes": "lambda_net", "input_setting": "Y", "input_mask": "Phi"},
    ]

    input_setting: Optional[str] = getattr(args, "input_setting", None)
    input_mask: Optional[str] = getattr(args, "input_mask", None)

    for config in configs:
        if config["name_includes"] in args.template:
            input_setting = config["input_setting"]
            input_mask = config["input_mask"]

    return input_setting, input_mask
