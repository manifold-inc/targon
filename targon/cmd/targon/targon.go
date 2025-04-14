package main

import (
	"targon/internal/setup"
	"targon/internal/targon"

	"github.com/subtrahend-labs/gobt/boilerplate"
)

func main() {
	deps := setup.Init()
	deps.Log.Infof("Starting validator with key [%s] on chain [%s] version [%d]", deps.Hotkey.Address, deps.Env.CHAIN_ENDPOINT, deps.Env.VERSION)

	core := targon.CreateCore(deps)
	validator := boilerplate.NewChainSubscriber(deps.Env.NETUID)
	deps.Log.Infof("Creating validator on netuid [%d]", validator.NetUID)

	targon.AddBlockCallbakcs(validator, core)
	targon.SetMainFunc(validator, core)

	validator.SetOnSubscriptionCreationError(func(e error) {
		deps.Log.Infow("Creation Error", "error", e)
		panic(e)
	})
	validator.SetOnSubscriptionError(func(e error) {
		deps.Log.Infow("Subscription Error", "error", e)
		panic(e)
	})
	validator.Start(deps.Client)
}
