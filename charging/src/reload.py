#credit: (https://stackoverflow.com/questions/15506971/recursive-version-of-reload)

import sys
import importlib
from types import ModuleType

def deep_reload(package_name: str):
    """
    Recursively reload all modules under the specified package.

    Parameters:
    - package_name (str): Name of the package to reload.
    """
    package_name_with_dot = package_name + '.'  # Append a dot for submodules
    modules_to_reload = [
        name for name in sys.modules
        if name.startswith(package_name_with_dot) or name == package_name
    ]
    # Reload modules from the deepest to the shallowest
    for module_name in sorted(modules_to_reload, key=lambda x: x.count('.'), reverse=True):
        try:
            module = sys.modules[module_name]
            if isinstance(module, ModuleType):  # Ensure it is a module
                importlib.reload(module)
                print(f"Reloaded: {module_name}")
        except Exception as e:
            print(f"Failed to reload: {module_name}. Error: {e}")
