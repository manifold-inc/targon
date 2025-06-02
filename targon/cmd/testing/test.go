package main

import (
	"fmt"
	"targon/internal/setup"
	"targon/internal/targon"
)

func main() {
	deps := setup.Init()
	fmt.Println(targon.NewNonce(deps.Hotkey.Address))
}
