from dataclasses import asdict
from datetime import datetime
import json
import os
import pathlib
import socket
from typing import Any
import warnings
import torch
import random

try:
    import pynvml

    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False


def select_best_device(mode="m", suppress_output=False):
    """
    Select the best available GPU device based on specified criteria.

    Args:
        mode (str): Selection mode - 'm' for most free memory, 'u' for least utilization. 'u' requires pynvml package to be installed.
        suppress_output (bool): If True, suppress all print and warning outputs

    Returns:
        torch.device: Selected device (GPU or CPU if no GPU available).

    Raises:
        Exception: If mode is not 'm' or 'u'.
    """
    if not torch.cuda.is_available():
        if not suppress_output:
            print("select_best_device(): Using CPU")
        return torch.device("cpu")

    if mode not in ["m", "u"]:
        raise Exception(
            f'select_device_with_most_free_memory: Acceptable inputs for mode are "m" (most free memory) and "u" (least utilization_). You specified: {mode}'
        )

    indices = list(range(torch.cuda.device_count()))
    random.shuffle(
        indices
    )  # shuffle the indices we iterate through so that, if, say, a bunch of processes scramble for GPUs at once, the first one won't get them all

    if mode == "m":
        max_free_memory = 0
        device_index = 0
        for i in indices:
            free_memory = torch.cuda.mem_get_info(i)[0]
            if free_memory > max_free_memory:
                max_free_memory = free_memory
                device_index = i
        return torch.device(f"cuda:{device_index}")

    elif mode == "u":
        if HAS_PYNVML:
            pynvml.nvmlInit()
            min_util = 100
            device_index = 0
            for i in indices:
                handle = pynvml.nvmlDeviceGetHandleByIndex(
                    i
                )  # Get the handle for the target GPU
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utilization = util.gpu  # GPU utilization percentage (integer)
                if gpu_utilization < min_util:
                    min_util = gpu_utilization
                    device_index = i
            pynvml.nvmlShutdown()

            # If all the GPUs are basically at max util, then make choice via memory availiability
            if min_util > 95:
                return select_best_device(mode="m")

            return torch.device(f"cuda:{device_index}")
        else:
            if not suppress_output:
                warnings.warn(
                    "Utilization 'u' based selection is only available if pnyvml is available, but it is not. Please install pnyvml to use mode 'u'. Switching to mode 'm' (memory-based device selection)"
                )
            return select_best_device("m")


def get_project_root() -> pathlib.Path:
    """
    Determines the project root using a fixed relative path from this file.
    It assumes this file is located within the 'project_root/src' directory.
    Validates the assumed root by checking for an 'src' directory within it.

    Returns:
        pathlib.Path: The absolute path to the project root directory.

    Raises:
        FileNotFoundError: If the 'src' directory is not found at the
                           assumed project root, indicating a potential
                           misconfiguration or change in directory structure.
        RuntimeError: If this utility file's location has changed such that
                      the fixed relative path logic is no longer valid.
    """
    try:
        # Get the absolute path of the current file (path_helpers.py)
        this_file_path = pathlib.Path(__file__).resolve()

        # Assumed structure: project_root/src/<this file>.py
        assumed_project_root = this_file_path.parent.parent
    except IndexError:
        # This would happen if Path(__file__).parent goes above filesystem root
        # which implies the file is not deep enough for this logic.
        raise RuntimeError(
            f"The utility file '{__file__}' seems to be located too high in the"
            f"directory tree for the fixed relative path logic to apply."
            f"Expected 'project_root/src/<this file>.py'."
            f"Instead got {this_file_path}."
        )

    # Validate: Check for the presence of an 'src' directory in the assumed root.
    # This 'src' directory is the one directly under the project_root.
    expected_src_dir = assumed_project_root / "src"

    if not (expected_src_dir.exists() and expected_src_dir.is_dir()):
        raise FileNotFoundError(
            f"Validation failed: An 'src' directory was not found at the assumed "
            f"project root '{assumed_project_root}'.\n"
            f"This function expects the project structure to be 'project_root/src/...', "
            f"and this utility file ('{__file__}') to be at a certain fixed location "
            f"within 'src/'. If the structure or file location has changed, "
            f"this function may need an update."
        )

    return assumed_project_root


def save_as_json(
    data, save_path: os.PathLike, override_if_exists: bool = False
) -> None:
    """Save arbitrary basic Python structures (primitives, lists, dicts, tuples) as JSON.

    Note: Tuples will be converted to lists when saved, as JSON doesn't have a tuple type.
    Sets will also be converted to lists. Other non-JSON-serializable types will raise an error.

    Args:
        data: The data to save (primitives, lists, dicts, tuples of primitives)
        save_path (os.PathLike): Path where to save the JSON file
        override_if_exists (bool): If False, raises error if file already exists. If True, overwrites.
    """
    save_path = pathlib.Path(save_path)

    if save_path.exists() and not override_if_exists:
        raise ValueError(
            f"File {save_path} already exists and override_if_exists is False"
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(data, f, indent=2)


def load_from_json(load_path: os.PathLike) -> Any:
    """Load data from JSON file.

    Returns the loaded data structure. Note that tuples saved as JSON will be loaded as lists, since JSON doesn't distinguish between lists and tuples.
    """
    with open(load_path, "r") as f:
        return json.load(f)


class JsonMixin:
    """Mixin class to add JSON save/load functionality to any dataclass"""

    def save_as_json(self, filepath: os.PathLike, override_if_exists: bool = False):
        """Save to JSON file

        Args:
            filepath (os.PathLike): Path where to save the JSON file
            override_if_exists (bool): If False, raises error if file already exists. If True, overwrites.
        """
        save_as_json(asdict(self), filepath, override_if_exists)

    @classmethod
    def load_from_json(cls, filepath: os.PathLike) -> "JsonMixin":
        """Load from JSON file"""
        data = load_from_json(filepath)
        return cls(**data)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str):
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls(**data)


def str_formatted_datetime() -> str:
    """
    Get current datetime as a formatted string.

    Returns:
        str: Datetime string in format 'YYYYMMDD_HHMMSS'.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_run_identifier() -> str:
    """
    Get a unique string to uniquely name a run/trial
    Returns:
        str: Datetime followed by hostname. Separated by hyphen "-"
    """
    return str_formatted_datetime() + "-" + socket.gethostname()
