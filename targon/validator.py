import copy
import bittensor as bt

import argparse
from targon import add_verifier_args, config, check_config, add_args

class Verifier:
    @classmethod
    def check_config(cls, config: "bt.Config"):
        check_config(cls, config)

    @classmethod
    def add_args(cls, parser):
        add_args(cls, parser)

    @classmethod
    def _config(cls):
        return config(cls)
    
    @property
    def block(self):
        return self.subtensor.get_current_block()
    
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        add_verifier_args(cls, parser)    

    def __init__(self, config=None):

        # setup config
        base_config = copy.deepcopy(config or self._config())
        self.config = self._config()
        self.config.merge(base_config)
        self.check_config(self.config)
        
        # setup logging
        bt.logging(config=self.config, logging_dir=self.config.full_path)

        bt.logging.info(self.config)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    Verifier.add_args(parser)
    args = parser.parse_args()
    verifier = Verifier(args)

