import sys
from pathlib import Path

# The app's modules import each other by bare name (the worker runs as a script
# from src/bgbg), so the unit tests import them the same way.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "bgbg"))
