"""Created on: May 2, 2023
Credit: Antonio Addis (IPAC)"""
import os
import yaml
import inspect
from pathlib import Path


class GsfConfig:

    def __init__(self) -> None:
        self.conf = None

    def load_configuration(self, configurationfile):
        configurationfile = GsfConfig._loadyaml(configurationfile)

        self.conf = configurationfile


    @staticmethod
    def _loadyaml(file):
        with open(file, 'r') as yamlfile:
            return yaml.safe_load(yamlfile)


    def getConf(self, key=None, subkey=None):
        if key and key in self.conf:
            if subkey and subkey in self.conf[key]:
                return self.conf[key][subkey]
            else:
                return self.conf[key]
        else:
            return self.conf
        
    @staticmethod
    def getConfigurationFile():
        
        class_path = Path(inspect.getfile(GsfConfig)).resolve().parent
        config_path = class_path.joinpath("config.yaml")

        if os.path.exists(config_path):
            return str(Path(config_path))
        else:
            raise FileNotFoundError("config.yaml not found")