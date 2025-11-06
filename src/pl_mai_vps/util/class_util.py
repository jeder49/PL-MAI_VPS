import importlib.util
import importlib
import sys
import os


def load_class_from_string(class_path, force_reload=False):
    """
    Load a class from a string like 'baseline.py/MyClass'
    Works on Windows, macOS, and Linux
    """
    if '/' not in class_path:
        raise ValueError("Format should be 'filename.py/ClassName'")

    file_path, class_name = class_path.rsplit('/', 1)

    # Normalize the file path for cross-platform compatibility
    file_path = os.path.normpath(file_path)

    # Get absolute path to avoid issues
    if not os.path.isabs(file_path):
        file_path = os.path.abspath(file_path)

    module_name = os.path.splitext(os.path.basename(file_path))[0]

    # Check if module is already loaded
    if module_name in sys.modules and not force_reload:
        module = sys.modules[module_name]
    else:
        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            raise ImportError(f"Could not load spec from {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # If we're force reloading an existing module
        if force_reload and module_name in sys.modules:
            importlib.reload(module)

    # Get the class
    if not hasattr(module, class_name):
        raise AttributeError(f"Class '{class_name}' not found in {file_path}")

    return getattr(module, class_name)
