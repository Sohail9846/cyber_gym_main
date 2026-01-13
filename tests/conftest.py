import sys
import os

# Ensure project root is on sys.path for imports like `import main`, `import vm_config`
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)