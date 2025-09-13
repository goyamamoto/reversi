import os
import sys


def _add_project_root_to_path():
    here = os.path.dirname(__file__)
    root = os.path.abspath(os.path.join(here, ".."))
    if root not in sys.path:
        sys.path.insert(0, root)


_add_project_root_to_path()

