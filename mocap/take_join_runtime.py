"""Command for the take_join helper; stdlib only, because the GUI imports it."""
import os
import sys

from .paths import ROOT


def command(args):
    if getattr(sys, 'frozen', False) or os.environ.get('RGB_MOCAP_MANAGED') == '1':
        return [str(ROOT/'runtime/python.exe'), '-I', '-X', 'utf8', '-B', '-u', str(ROOT/'take_join_boot.py'), *map(str, args)]
    return [sys.executable, '-X', 'utf8', '-B', '-u', '-m', 'mocap.take_join', *map(str, args)]
