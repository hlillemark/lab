# python/print_env.py
import sys
import os

print("Python executable:", sys.executable)
print("sys.path:", sys.path)
print("env PATH:", os.environ.get("PATH"))
print("env LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
