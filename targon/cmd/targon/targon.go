package main

import (
	"context"
	"os"
	"os/signal"
	"syscall"
	"time"

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
		deps.Env.ChainEndpoint,
		deps.Env.Version,
	)
	if deps.Mongo != nil {
		defer func() {
			if err := deps.Mongo.Disconnect(context.Background()); err != nil {
				deps.Log.Errorw("failed disconnecting from mongo", "error", err)
			}
		}()
	}

	core := targon.CreateCore(deps)
	validator := boilerplate.NewChainSubscriber()
	deps.Log.Infof("Creating validator on netuid [%d]", deps.Env.Netuid)

	callbacks.AddBlockCallbacks(validator, core)

	validator.SetOnSubscriptionError(func(e error) {
		deps.Log.Errorw("Subscription Error", "error", e)
	})
	err := targon.LoadMongoBackup(core)
	if err != nil {
		core.Deps.Log.Warn("Failed to load last checkpoint")
	}
	if err == nil {
		core.Deps.Log.Info("Loaded checkpoint from mongo")
	}
	go func() {
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
		<-sigChan
		validator.Stop()
	}()

	for {
		err := validator.Start(deps.Client)
		if err != nil {
			deps.Log.Errorw("Subscription Error", "error", err)
			time.Sleep(5 * time.Second)
			continue
		}
		break
	}
	core.Deps.Log.Info("Shutting down validator")
	err = targon.SaveMongoBackup(core)
	if err != nil {
		core.Deps.Log.Errorw("Failed saving backup of state", "error", err)
	}
}
