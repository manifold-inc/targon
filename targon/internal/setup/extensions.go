package setup

import (
	"github.com/centrifuge/go-substrate-rpc-client/v4/types/extrinsic"
	"github.com/centrifuge/go-substrate-rpc-client/v4/types/extrinsic/extensions"
)

// The finney runtime added transaction extensions that gobt does not register,
// so signing any extrinsic (e.g. serve_axon, set_weights) fails with
// "signed extension type not supported". These extensions carry no signed or
// implicit payload data, so register them as no-op mutators.
func init() {
	noop := func(payload *extrinsic.Payload) {}
	for _, name := range []string{
		"SudoTransactionExtension",
		"CheckShieldedTxValidity",
	} {
		if _, ok := extrinsic.PayloadMutatorFns[extensions.SignedExtensionName(name)]; !ok {
			extrinsic.PayloadMutatorFns[extensions.SignedExtensionName(name)] = noop
		}
	}
}
