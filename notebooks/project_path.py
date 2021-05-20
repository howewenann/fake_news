import sys
import os

def project_path(level):
    depth = ['..'] * level
    depth = '/'.join(depth)
    module_path = os.path.abspath(depth)
    print(module_path)
    if module_path not in sys.path:
        sys.path.append(module_path)