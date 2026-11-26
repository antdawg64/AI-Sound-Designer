"""Run a Modal command from a cloud session: `python scripts/modal_cmd.py run modal_library.py`.
Same as the `modal` command, with the proxy certificate fix applied first."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soundlib import proxyfix
proxyfix.apply()
from modal.__main__ import main
sys.argv = ["modal", *sys.argv[1:]]
main()
