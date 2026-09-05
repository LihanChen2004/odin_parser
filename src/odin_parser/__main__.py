"""Allow ``python -m odin_parser`` to invoke the CLI."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
