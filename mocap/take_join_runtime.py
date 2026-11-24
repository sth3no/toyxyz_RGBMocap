"""Commands for the take_join and take_import helpers; stdlib only, because the GUI imports it."""
import os
import sys

from .paths import ROOT


def command(args, module='take_join'):
    if getattr(sys, 'frozen', False) or os.environ.get('RGB_MOCAP_MANAGED') == '1':
        return [str(ROOT/'runtime/python.exe'), '-I', '-X', 'utf8', '-B', '-u', str(ROOT/f'{module}_boot.py'), *map(str, args)]
    return [sys.executable, '-X', 'utf8', '-B', '-u', '-m', f'mocap.{module}', *map(str, args)]


def import_command(args):
    return command(args, 'take_import')
