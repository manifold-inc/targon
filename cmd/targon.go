package main

import (
	"fmt"
	"strconv"

	"targon/internal/setup"
	"targon/internal/subtensor/utils"

	"github.com/centrifuge/go-substrate-rpc-client/v4/types"
	"github.com/subtrahend-labs/gobt/runtime"
)

func main() {
	deps := setup.Init()
	deps.Log.Infof("Starting validator with key [%s] on chain [%s]", deps.Env.HOTKEY_SS58, deps.Env.CHAIN_ENDPOINT)

	blockHash, err := deps.Client.Api.RPC.Chain.GetBlockHashLatest()
	if err != nil {
		deps.Log.Fatalf("Error getting latest block hash: %s", err)
	}

	netuid := 4
	deps.Log.Infof("Testing netuid [%d]", netuid)
	neurons, err := runtime.GetNeurons(deps.Client, uint16(netuid), &blockHash)
	if err != nil {
		deps.Log.Infof("Error fetching neurons for netuid [%d]: [%s]", netuid, err)
	}

	if len(neurons) == 0 {
		deps.Log.Infof("No neurons found for netuid [%d]", netuid)
	}

	deps.Log.Infof("total of %d neurons", len(neurons))
	for i, neuron := range neurons {
		deps.Log.Infow(
			"neuron",
			"Neuron",
			strconv.Itoa(i),
			"Hotkey",
			utils.AccountIDToSS58(neuron.Hotkey),
			"Coldkey",
			utils.AccountIDToSS58(neuron.Coldkey),
			"UID",
			fmt.Sprintf("%d", neuron.UID.Int64()),
			"Active",
			fmt.Sprintf("%v", neuron.Active == types.NewBool(true)),
		)
	}
	deps.Log.Sync()
}
