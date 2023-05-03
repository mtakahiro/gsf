"""Created on: May 2, 2023
Credit: Antonio Addis (IPAC)"""
import sys
import logging
from pathlib import Path
from time import strftime
from os.path import expandvars
from colorama import Fore, Back, Style

class Singleton(type):
    '''Make sure there is a single instance of the logger class at any time.'''
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(
                Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
    
       
# https://gist.github.com/joshbode/58fac7ababc700f51e2a9ecdebe563ad
class ColoredFormatter(logging.Formatter):

    def __init__(self, *args, colors=None, **kwargs):
        """Initialize the formatter with specified format strings."""

        super().__init__(*args, **kwargs)

        self.colors = colors if colors else {}

    def format(self, record) -> str:
        """Format the specified record as text."""

        record.color = self.colors.get(record.levelname, '')
        record.reset = Style.RESET_ALL

        return super().format(record)




class GsfLogger(metaclass=Singleton):
    '''This class defines the logging format, level and output.'''

    def __init__(self):
        self.formatter = ColoredFormatter(
            '{asctime} |{color} {levelname:8} {reset}| {name} | {message} | ({filename}:{lineno})',
            style='{', datefmt='%Y-%m-%d %H:%M:%S',
            colors={
                'DEBUG': Fore.CYAN,
                'INFO': Fore.GREEN,
                'WARNING': Fore.YELLOW,
                'ERROR': Fore.RED,
                'CRITICAL': Fore.RED + Back.WHITE + Style.BRIGHT,
            }
        )
        self.logLevel = None
        self.rootLogsDir = None
        self.today = strftime('%Y%m%d')
        self.now = strftime('%H%M%S')

    @staticmethod
    def mapLogLevel(verboseLvl):
        if verboseLvl == 0:
            return logging.WARNING
        elif verboseLvl == 1:
            return logging.INFO
        elif verboseLvl == 2:
            return logging.DEBUG
        else:
            raise ValueError(f"Invalid value for verboseLvl ({verboseLvl}). Allowed values are 0 (WARNING), 1 (INFO), 2 (DEBUG)")


    def setLogger(self, rootPath=None, logLevel=3):
        """
        Must be called once, before getLogger()
        It sets the root directory of the logs, the log level and the formatter. 
        It defines a common stream handler for all the loggers. (TODO: think about it in a multiprocessing/multithread context)
        """
        if rootPath is not None:
            self.rootLogsDir = Path(expandvars(rootPath)).joinpath("logs")
            self.rootLogsDir.mkdir(parents=True, exist_ok=True)
        
        self.logLevel = GsfLogger.mapLogLevel(logLevel)
        
        self.sh = logging.StreamHandler(sys.stdout)
        self.sh.setLevel(self.logLevel)
        self.sh.setFormatter(self.formatter)

        print(f"Log level set to {logging.getLevelName(self.logLevel)} and output to {self.rootLogsDir}")
        return self

    def getRootLogsDir(self):
        return self.rootLogsDir

    def getLogger(self, loggerName, id=None, addFileHandler=True):
        """
        We expect the loggerName to have the following format: <class_name>_<id>
        """
        if id is not None:
            logger = logging.getLogger(f"{loggerName}.{id}")
        else:
            logger = logging.getLogger(loggerName)

        logger.setLevel(self.logLevel)

        # avoid duplicating handlers   
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            logger.addHandler(self.sh)

        # We want each class to log in its own file.
        # We don't want to propagate the log events to the ancestors of the logger.
        logger.propagate = False


        if addFileHandler:

            if not self.rootLogsDir.exists():
                raise Exception(f"Log output directory {self.rootLogsDir} does not exist. Call setLogger() first.")

            if id is not None:
                loggerName += f"_{id}"

            loggerOutputDir = str(self.rootLogsDir.joinpath(loggerName+".log"))

            fh = logging.FileHandler(loggerOutputDir)
                
            fh.setLevel(self.logLevel)
            fh.setFormatter(self.formatter)

            # avoid duplicating handlers 
            if fh in logger.handlers:
                raise Exception(f"Logger already has a file handler {fh}")


            logger.addHandler(fh)

        return logger
    
    @staticmethod
    def getDefaultLogger(loggerName, logLevel):
        return GsfLogger() \
                        .setLogger(rootPath=None, logLevel=logLevel) \
                        .getLogger(loggerName, addFileHandler=False)