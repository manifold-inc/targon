
import bittensor as bt

'''
This is a blacklist module for validators. It is used to blacklist bad faith keys from the threat-actor file.
'''

def blacklist( self ):
    self.blacklisted_coldkeys = []
    with open('../blacklist.txt'):
        for line in blacklist:
            self.blacklisted_coldkeys.append(line)
            bt.logging.info('blacklisting low integrity key', line)

