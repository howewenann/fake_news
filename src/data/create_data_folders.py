# -*- coding: utf-8 -*-
import os
from pathlib import Path

def main():
    """ 
    Creates all data folders
    """
    proj_dir = Path.cwd().parents[1]
    Path(proj_dir, 'data', 'raw').mkdir(parents=True, exist_ok=True)
    Path(proj_dir, 'data', 'interim').mkdir(parents=True, exist_ok=True)
    Path(proj_dir, 'data', 'processed').mkdir(parents=True, exist_ok=True)
    Path(proj_dir, 'data', 'external').mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    main()
