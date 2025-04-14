package main

import (
	"targon/targon/internal/setup"
)

func main() {
	deps := setup.Init()
	deps.Log.Infof("Starting validator with key [%s]", deps.Env.HOTKEY_SS58)
}
