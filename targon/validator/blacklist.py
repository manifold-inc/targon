import os 
import bittensor as bt

'''
This is a blacklist module for validators. It is used to blacklist bad faith keys from the threat-actor file.
'''

def blacklist( self ):
    self.blacklisted_coldkeys = []
    # get the directory the file is in
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # step backwards one directory
    dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))

    with open(os.path.join(dir_path, 'threat-actors.txt'), 'r') as blacklist:
        for line in blacklist:
            self.blacklisted_coldkeys.append(line)
            bt.logging.info('blacklisting low integrity key', line)

