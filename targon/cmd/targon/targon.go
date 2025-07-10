package main

import (
	"context"

	"targon/internal/callbacks"
	"targon/internal/setup"
	"targon/internal/targon"

	"github.com/subtrahend-labs/gobt/boilerplate"
)

func main() {
	deps := setup.Init()
	deps.Log.Infof(
		"Starting validator with key [%s] on chain [%s] version [%d]",
		deps.Hotkey.Address,
		deps.Env.CHAIN_ENDPOINT,
		deps.Env.VERSION,
	)
	if deps.Mongo != nil {
		defer func() {
			if err := deps.Mongo.Disconnect(context.Background()); err != nil {
				deps.Log.Errorw("failed disconnecting from mongo", "error", err)
			}
		}()
	}

	core := targon.CreateCore(deps)
	validator := boilerplate.NewChainSubscriber(deps.Env.NETUID)
	deps.Log.Infof("Creating validator on netuid [%d]", validator.NetUID)

	callbacks.AddBlockCallbacks(validator, core)
	targon.SetMainFunc(validator, core)

	validator.SetOnSubscriptionCreationError(func(e error) {
		deps.Log.Infow("Failed to connect to chain", "error", e)
		panic(e)
	})
	validator.SetOnSubscriptionError(func(e error) {
		deps.Log.Infow("Subscription Error", "error", e)
	})
	err := targon.LoadMongoBackup(core)
	if err != nil {
		core.Deps.Log.Warn("Failed to load last checkpoint")
	}
	if err == nil {
		core.Deps.Log.Info("Loaded checkpoint from mongo")
	}
	validator.Start(deps.Client)
}
