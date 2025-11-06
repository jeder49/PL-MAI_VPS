import importlib.util
import importlib
import sys
import os
from pathlib import Path


def load_class_from_string(base_path, class_path, force_reload=False):
    """
    Load a class from a string like 'baseline.py/MyClass'
    base_path: Path(__file__) or similar - used as reference point
    class_path: 'filename.py/ClassName' - relative to base_path's parent
    """
    if '/' not in class_path:
        raise ValueError("Format should be 'filename.py/ClassName'")

    file_path, class_name = class_path.rsplit('/', 1)

    # Convert base_path to Path object and get parent directory
    base_dir = Path(base_path).parent

    # Resolve the full path relative to base directory
    full_file_path = (base_dir / file_path).resolve()

    module_name = full_file_path.stem

    # Check if module is already loaded
    if module_name in sys.modules and not force_reload:
        module = sys.modules[module_name]
    else:
        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, str(full_file_path))
        if spec is None:
            raise ImportError(f"Could not load spec from {full_file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # If we're force reloading an existing module
        if force_reload and module_name in sys.modules:
            importlib.reload(module)

    # Get the class
    if not hasattr(module, class_name):
        raise AttributeError(f"Class '{class_name}' not found in {full_file_path}")

    return getattr(module, class_name)
