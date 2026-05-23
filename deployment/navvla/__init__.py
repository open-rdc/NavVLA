"""Source package for NavVLA deployment nodes."""
import os
import sys

_omnivla_parent = os.path.normpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
)
if _omnivla_parent not in sys.path:
    sys.path.insert(0, _omnivla_parent)
