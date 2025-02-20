import re
import sys
import os
from pkg_resources import get_distribution, DistributionNotFound

__version_commit__ = ''
_regex_git_hash = re.compile(r'.*\+g(\w+)')

from .version import __version__
# try:
#     __version__ = get_distribution(__name__).version
# except DistributionNotFound:
#     __version__ = 'dev'

if '+' in __version__:
    commit = _regex_git_hash.match(__version__).groups()
    if commit:
        __version_commit__ = commit[0]

__author__ = 'Takahiro Morishita'
__email__ = 'takahiro@ipac.caltech.edu'
__credits__ = 'IPAC'

package = 'gsf'

print('Welcome to %s version %s'%(package,__version__))

# Add path;
#sys.path.append('%stmphot/'%borgpipe)

import matplotlib as mat
mat.rcParams['font.family'] = 'StixGeneral'
