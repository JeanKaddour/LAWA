import os
from pathlib import Path

cwd = Path(".").resolve()
print("running 'check_installation.py' from:", cwd)

# Old versions of numpy/torch can prevent loading the .so files
import torch

print("torch:", torch.__version__)
import numpy

print("numpy:", numpy.__version__)

import fairseq

print("Fairseq installed at:", fairseq.__file__)
import _imp
import fairseq.criterions
import fairseq.dataclass.configs

print("Should load following .so suffixes:", _imp.extension_suffixes())

so_files = list(Path(fairseq.__file__).parent.glob("*.so"))
so_files.extend(Path(fairseq.__file__).parent.glob("data/*.so"))
print("Found following .so files:")
for so_file in so_files:
    print(f"- {so_file}")

from fairseq import libbleu

print("Found libbleu at", libbleu.__file__)
from fairseq.data import data_utils_fast

print("Found data_utils_fast at", data_utils_fast.__file__)
