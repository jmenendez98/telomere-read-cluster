"""`python -m trc` runs the same CLI the `trc` console script does."""
import sys

from .main import main

if __name__ == "__main__":
    sys.exit(main())
