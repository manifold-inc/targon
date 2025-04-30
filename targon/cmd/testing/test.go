package main

import (
	"fmt"

	"targon/internal/subtensor/utils"

	"github.com/subtrahend-labs/gobt/client"
	"github.com/subtrahend-labs/gobt/runtime"
)

func main() {
	c, _ := client.NewClient("wss://test.finney.opentensor.ai:443")
	// c, _ := client.NewClient("wss://entrypoint-finney.opentensor.ai:443")
	blockHash, err := c.Api.RPC.Chain.GetBlockHashLatest()
	if err != nil {
		panic(err)
	}
	neurons, err := runtime.GetNeurons(c, uint16(337), &blockHash)
	if err != nil {
		panic(err)
	}
	for i, n := range neurons {
		fmt.Printf("%d: %s\n", i, utils.AccountIDToSS58(n.Hotkey))
	}
}
