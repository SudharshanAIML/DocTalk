import sys
import traceback

try:
    import main
    print("Import successful")
except Exception:
    traceback.print_exc()
