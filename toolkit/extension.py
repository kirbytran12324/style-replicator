import os
import importlib
import pkgutil
from typing import List

from toolkit.paths import TOOLKIT_ROOT


class Extension(object):
    """Base class for extensions."""

    name: str = None
    uid: str = None

    @classmethod
    def get_process(cls):
        # extend in subclass
        pass


def get_all_extensions() -> List[Extension]:
    """
    Discover extensions under 'extensions' and 'extensions_built_in'.
    Skips modules that fail to import, unless AITOOLKIT_STRICT_EXTENSIONS=1.
    Also falls back to scanning for subclasses of Extension if
    AI_TOOLKIT_EXTENSIONS is missing.
    """
    extension_folders = ['extensions', 'extensions_built_in']
    all_extension_classes: List[Extension] = []
    strict = os.environ.get("AITOOLKIT_STRICT_EXTENSIONS", "0") == "1"

    for sub_dir in extension_folders:
        extensions_dir = os.path.join(TOOLKIT_ROOT, sub_dir)
        if not os.path.isdir(extensions_dir):
            continue

        for (_, name, _) in pkgutil.iter_modules([extensions_dir]):
            pkg_name = f"{sub_dir}.{name}"
            try:
                module = importlib.import_module(pkg_name)
            except Exception as e:
                print(f"[extensions] Skipping '{pkg_name}' due to import error: {e}")
                if strict:
                    raise
                continue

            try:
                extensions = getattr(module, "AI_TOOLKIT_EXTENSIONS", None)
                if isinstance(extensions, list):
                    all_extension_classes.extend(extensions)
                else:
                    # Fallback: auto-discover classes that subclass Extension
                    for obj in vars(module).values():
                        try:
                            if isinstance(obj, type) and issubclass(obj, Extension) and obj is not Extension:
                                all_extension_classes.append(obj)
                        except Exception:
                            # Non-class or issubclass on non-type
                            pass
            except Exception as e:
                print(f"[extensions] Loaded '{pkg_name}' but failed to collect extensions: {e}")
                if strict:
                    raise
                continue

    return all_extension_classes


def get_all_extensions_process_dict():
    """
    Build a uid\->process map, skipping extensions that fail to provide a process.
    """
    all_extensions = get_all_extensions()
    process_dict = {}
    for extension in all_extensions:
        try:
            uid = getattr(extension, "uid", None)
            if not uid:
                print(f"[extensions] Skipping extension without uid: {extension}")
                continue
            process = extension.get_process()
            if process is None:
                print(f"[extensions] Skipping '{uid}' with no process.")
                continue
            process_dict[uid] = process
        except Exception as e:
            print(f"[extensions] Skipping '{getattr(extension, 'uid', extension)}' due to get_process error: {e}")
            continue
    return process_dict