import os
import time
import argparse
import threading
import bittensor as bt
from retry import retry

from substrateinterface.base import QueryMapResult, SubstrateInterface

class WalletRegister:
    def __init__(self, wallet_name=None, hotkey_name=None, netuid=1, n=0, snipe=False, watching=False):
        self.snipe = snipe
        self.netuid = netuid if netuid is not None else 4
        self.n = n if n is not None else 0
        self.RAOPERTAO = 1e9
        self.U64_MAX = 18_446_744_073_709_551_615
        self.wallet_name = wallet_name if wallet_name is not None else "lilith"
        self.subtensor = bt.subtensor( network="finney" )
        self.metagraph = self.subtensor.metagraph( netuid=self.netuid )
        self.wallets = {}
        self.extrinsics = []
        self.reloaded = False
        self.unlocked_coldkey = None
        self.watching = watching
        self.last_sync_block = None
        self.warmed_up = False
        self.refreshed = False

        if self.watching:
            bt.logging.info(f"!!!! NOT REGISTERING, JUST WATCHING !!!!")


    def loadWallets(self, hotkey_name):
        # Instantiate wallet for the current hotkey.
        wallet = bt.wallet(name=self.wallet_name, hotkey=hotkey_name,)
        ss58_address = wallet.hotkey.ss58_address

        if self.unlocked_coldkey is None:
            self.unlocked_coldkey = wallet.coldkey
        if ss58_address not in self.metagraph.hotkeys:
            print(wallet)
            self.wallets[hotkey_name] = wallet
            return wallet
        else:
            bt.logging.warning(f"Wallet {hotkey_name} did not pass is_registered() check. Skipping...")
            return None

    def getExtrinsics(self):
        extrinsics = []
        wallet_count = 0

        for key_name in os.listdir(os.path.expanduser(f"~/.bittensor/wallets/{self.wallet_name}/hotkeys")):
            wallet = self.loadWallets(key_name)
            if wallet is not None:
                # Compose a call for the specific wallet
                call = self.subtensor.substrate.compose_call(
                    call_module='SubtensorModule',
                    call_function='burned_register',
                    call_params={
                        'netuid': self.netuid,  # replace with the netuid you want to register on
                        'hotkey': wallet.hotkey.ss58_address
                    }
                )

                # Get the nonce for the account
                nonce = self.subtensor.substrate.get_account_nonce(self.unlocked_coldkey.ss58_address) + wallet_count

                # Create the signed extrinsic with the specific call
                extrinsic = self.subtensor.substrate.create_signed_extrinsic(call=call, keypair=self.unlocked_coldkey, nonce=nonce)

                # Append the extrinsic to the list
                extrinsics.append(extrinsic)


                if wallet_count >= 5:
                    break
                wallet_count += 1

        self.extrinsics = extrinsics

        bt.logging.info(f"Loaded {len(self.extrinsics)} extrinsics.")



    def getBurnRegistrationsThisInterval(self) -> int:
        with self.subtensor.substrate as substrate:
            return substrate.query('SubtensorModule', 'BurnRegistrationsThisInterval', [self.netuid]).value

    def getBurn(self) -> int:
        with self.subtensor.substrate as substrate:
            return substrate.query('SubtensorModule', 'Burn', [self.netuid]).value/self.RAOPERTAO

    def getTargetRegistrationsPerInterval(self) -> int:
        with self.subtensor.substrate as substrate:
            return substrate.query('SubtensorModule', 'TargetRegistrationsPerInterval', [self.netuid]).value

    def getLastAdjustmentBlock(self) -> int:
        with self.subtensor.substrate as substrate:
            return substrate.query('SubtensorModule', 'LastAdjustmentBlock', [self.netuid]).value
    

    def getAdjustmentInterval(self) -> int:
        with self.subtensor.substrate as substrate:
            return substrate.query('SubtensorModule', 'AdjustmentInterval', [self.netuid]).value

    def submitExtrinsic(self, extrinsic):
        self.subtensor.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=False, wait_for_finalization=False)
        bt.logging.info(f"Submitted extrinsic")

    def getAlpha(self):
        with self.subtensor.substrate as substrate:
            return substrate.query('SubtensorModule', 'AdjustmentAlpha', [self.netuid]).value

    def predictNextBurnCost(self, burn, registration_this_interval):
        # Get the alpha value
        alpha = self.getAlpha()/self.U64_MAX

        # Get the target registrations per interval
        target_registrations = self.getTargetRegistrationsPerInterval()

        # Compute the updated burn
        updated_burn = burn * (registration_this_interval + target_registrations) / (2 * target_registrations)

        # Compute the next burn value
        next_value = alpha * burn + (1 - alpha) * updated_burn

        return next_value

    def testLatency(self):
        last_adjustment_block = self.getLastAdjustmentBlock()
        current_block = self.subtensor.get_current_block()
        adjustment_interval = self.getAdjustmentInterval()
        next_adjustment_block = (last_adjustment_block + adjustment_interval) - current_block


        while next_adjustment_block > 0:
            time.sleep(0.1)
            current_block = self.subtensor.get_current_block()
            next_adjustment_block = (last_adjustment_block + adjustment_interval) - current_block
            print(current_block)

    def getVariables(self):
        burn = self.getBurn()
        registration_this_interval = self.getBurnRegistrationsThisInterval()
        last_adjustment_block = self.getLastAdjustmentBlock()
        current_block = self.subtensor.get_current_block()
        adjustment_interval = self.getAdjustmentInterval()

        next_adjustment_block = (last_adjustment_block + adjustment_interval) - current_block
        next_predicted_burn = self.predictNextBurnCost(burn, registration_this_interval)
        return burn, registration_this_interval, last_adjustment_block, current_block, adjustment_interval, next_adjustment_block, next_predicted_burn
    
    def run(self):
        burn, registration_this_interval, last_adjustment_block, current_block, adjustment_interval, next_adjustment_block, next_predicted_burn = self.getVariables()
        print ( f"burn: {burn}, registration_this_interval: {registration_this_interval}, last_adjustment_block: {last_adjustment_block}, current_block: {current_block}, next_adjustment_block: {next_adjustment_block}, next_predicted_burn: {next_predicted_burn}" )

        
        # Check if it is within 2 blocks of a predicted change in the burn value
        if next_adjustment_block <= 20 and not self.watching and not self.refreshed:
            self.metagraph = self.subtensor.metagraph( netuid=self.netuid )
            self.getExtrinsics()
            self.refreshed = True

        if next_adjustment_block <= 2 and not self.watching and self.refreshed:
            bt.logging.info(f"Predicted burn {next_predicted_burn} is less than current burn {burn}. Registering wallets‼️‼️‼️")
            # Wait until 2-3 seconds before the next adjusted interval block
            while next_adjustment_block > 1 and registration_this_interval == self.getTargetRegistrationsPerInterval():
                time.sleep(0.1)
                current_block = self.subtensor.get_current_block()
                next_adjustment_block = (last_adjustment_block + adjustment_interval) - current_block
                bt.logging.info(f"current_block", current_block)
            bt.logging.info(f"Submitting extrinsics at block {current_block}‼️‼️‼️")
            time.sleep( 10 )
            bt.logging.info(f"Submitting extrinsics at block {current_block}‼️‼️‼️")
            # Submit transactions to register 6 wallets at the predicted adjustment block
            all_threads = []
            
            for extrinsic in self.extrinsics:
                self.submitExtrinsic(extrinsic)
                print("Submitted extrinsic")
            
                time.sleep( 12 )
            # time.sleep( 12 )
            # self.metagraph.sync( lite=False )
            # self.getExtrinsics()
            burn, registration_this_interval, last_adjustment_block, current_block, adjustment_interval, next_adjustment_block, next_predicted_burn = self.getVariables()
            print(f"burn: {burn}, registration_this_interval: {registration_this_interval}, last_adjustment_block: {last_adjustment_block}, current_block: {current_block}, next_adjustment_block: {next_adjustment_block}, next_predicted_burn: {next_predicted_burn}")
            self.refreshed = False
            self.extrinsics = []
        
        if self.snipe and not self.watching and registration_this_interval < (self.getTargetRegistrationsPerInterval()*3) and self.refreshed:
            for extrinsic in self.extrinsics:
                self.submitExtrinsic(extrinsic)
                print("Submitted extrinsic")
            
                time.sleep( 12 )

            # self.metagraph.sync( lite=False )
            # self.getExtrinsics()
            burn, registration_this_interval, last_adjustment_block, current_block, adjustment_interval, next_adjustment_block, next_predicted_burn = self.getVariables()
            print(f"burn: {burn}, registration_this_interval: {registration_this_interval}, last_adjustment_block: {last_adjustment_block}, current_block: {current_block}, next_adjustment_block: {next_adjustment_block}, next_predicted_burn: {next_predicted_burn}")
            self.refreshed = False
            self.extrinsics = []

        time.sleep( 3 )

parser = argparse.ArgumentParser(description='Register wallets on the Substrate network.')
parser.add_argument('--instance_n', type=int, default=0, help='Number of wallets to register')
parser.add_argument('--snipe', action='store_true', help='Snipe the next burn adjustment')
parser.add_argument('--netuid', type=int, default=1, help='Network UID')
parser.add_argument('--wallet_name', type=str, default="lilith", help='Wallet name')
parser.add_argument('--watcher', help='Watch for updates, no reg', action='store_true')

config = bt.config( parser )

wallet_register = WalletRegister(n=config.instance_n, snipe=config.snipe, wallet_name=config.wallet_name, netuid=config.netuid, watching=config.watcher)
# wallet_register.testLatency()
if not config.watcher:
    wallet_register.getExtrinsics()
    wallet_register.refreshed = True
while True:
    try:
        wallet_register.run()
    except KeyboardInterrupt:
        break