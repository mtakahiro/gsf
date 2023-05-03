"""Created on: May 2, 2023
Credit: Antonio Addis (IPAC)"""
import os
from pathlib import Path
from shutil import rmtree
import ast
import json

from .GsfConfig import GsfConfig
from .GsfLogger import GsfLogger

class GsfBase:
    """
    Base class used for GSF common variables/functions/utilities (Logger, configfile parsing etc)

    """

    def __init__(self, configurationfile) -> None:
        """
        Args:
        configurationFile(str): a relative or absolute path to the yaml configuration file.
        
        """
        self.gsfconfig = GsfConfig()

        self.gsfconfig.load_configuration(os.path.expandvars(configurationfile))

        # self.inputdir = os.path.expandvars(self.romanconfig.getConf("input","input_data"))

        # self.outdir = os.path.expandvars(self.romanconfig.getConf("output","outdir"))

        self.logdir = './'

        # Path(self.outdir).mkdir(parents=True, exist_ok=True)
        # Path(self.inputdir).mkdir(parents=True, exist_ok=True)

        self.logger = GsfLogger()
        self.logger.setLogger(self.logdir, self.gsfconfig.getConf("output","verboselvl"))

    def getAnalysisDir(self):
        if self.outdir.exists() and self.outdir.is_dir():
            return str(self.outdir)
        
        else:
            print("OutputDirectory not found")

    
    def deleteAnalysisDir(self):

        if self.outdir.exists() and self.outdir.is_dir():
            rmtree(self.outdir)
        else:
            return False

        return True

    def load_json(self, input_file):

        roman_env = os.getenv("GSF")
    
        with open(input_file, 'r') as f:
            listobject = json.load(f)
        
        listobject_str = str(listobject).replace("$GSF", roman_env)
        
        listobject_dict = ast.literal_eval(listobject_str)
       
        return listobject_dict