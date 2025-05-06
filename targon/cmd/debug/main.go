package main

import (
	"targon/internal/setup"
)

func main() {
	deps := setup.Init()
	deps.Log.Infof(
		"Starting validator with key [%s] on chain [%s] version [%d]",
		deps.Hotkey.Address,
		deps.Env.CHAIN_ENDPOINT,
		deps.Env.VERSION,
	)

	//core := targon.CreateCore(deps)
}
