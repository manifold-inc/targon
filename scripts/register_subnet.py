import bittensor as bt
import time

wallet = bt.wallet( name="lilith" )
subtensor = bt.subtensor( network="finney" )
# unlock the wallet
balance = subtensor.get_balance( wallet.coldkeypub.ss58_address )

burn_cost = bt.utils.balance.Balance( subtensor.get_subnet_burn_cost() )

unlocked_wallet = wallet.coldkey

tries = 1000

if burn_cost > balance:
    raise ValueError( f"Insufficient balance: {balance} < {burn_cost} you lose!" )

# bt.logging.info( f"Balance: {balance} > {burn_cost} you win!" )
def create_extrinsic( ):
    with subtensor.substrate as substrate:
        call = substrate.compose_call(
            call_module="SubtensorModule",
            call_function="register_network",
            call_params={
                'immunity_period': 0,
                'reg_allowed': True,
            },
        )

        extrinsic = substrate.create_signed_extrinsic(
            call=call,
            keypair=unlocked_wallet,
        )

        return extrinsic

def submit_extrinsic( extrinsic ):
    with bt.__console__.status(":satellite: Registering subnet..."):
        with subtensor.substrate as substrate:
            receipt = substrate.submit_extrinsic( extrinsic, wait_for_inclusion=True, wait_for_finalization=True )
            print(receipt)
            # process if registration successful
            receipt.process_events()
            bt.logging.info( receipt )
            if not receipt.is_success:
                bt.__console__.print(
                    ":cross_mark: [red]Failed[/red]: error:{}".format(
                        receipt.error_message
                    )
                )
                time.sleep(0.5)


            # Successful registration, final check for membership
            else:
                bt.__console__.print(
                    f":white_heavy_check_mark: [green]Registered subnetwork with netuid: {receipt.triggered_events[1].value['event']['attributes'][0]}[/green]"
                )
                return True

for attempt in range( tries ):
    start_time = time.time()
    signed_extrinisc = create_extrinsic()
    end_time = time.time() - start_time

    bt.logging.info( f"Extrinsic created in {end_time} seconds" )

    bt.logging.info( f"Sending extrinsic" )

    start_send_time = time.time()
    success = submit_extrinsic( signed_extrinisc )

    if success:
        end_send_time = time.time() - start_send_time
        bt.logging.info( f"Extrinsic sent in {end_send_time} seconds" )
        break

    else:
        bt.logging.info('Extrinsic failed, trying again')
        continue
