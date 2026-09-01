import re
#import sys
#import os
# from pkg_resources import get_distribution, DistributionNotFound
from .version import __version__
__version_commit__ = ""
_regex_git_hash = re.compile(r".*\+g(\w+)")

if '+' in __version__:
    commit = _regex_git_hash.match(__version__).groups()
    if commit:
        __version_commit__ = commit[0]

from importlib.metadata import version
__version_commit__ = version("gsf")

__author__ = "Takahiro Morishita"
__email__ = "morishita@astr.tohoku.ac.jp"
__credits__ = "Tohoku University"

package = 'gsf'

print('Welcome to %s version %s'%(package,__version__))

# Add path;
#sys.path.append('%stmphot/'%borgpipe)

import matplotlib as mat
mat.rcParams['font.family'] = 'StixGeneral'
