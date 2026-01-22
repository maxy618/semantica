import sys
import os
import multiprocessing
from termcolor import cprint

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from semantica.main import main
except ImportError:
    sys.path.insert(0, os.path.abspath('src'))
    from semantica.main import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        cprint(f"Critical error: {e}", "red")
        sys.exit(1)