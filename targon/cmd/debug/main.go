package main

import (
	"flag"
	"fmt"
	"net/http"
	"time"

	"targon/internal/setup"
	"targon/internal/targon"

	"github.com/subtrahend-labs/gobt/runtime"
)

func main() {
	deps := setup.Init()
	deps.Log.Infof(
		"Starting validator with key [%s] on chain [%s] version [%d]",
		deps.Hotkey.Address,
		deps.Env.CHAIN_ENDPOINT,
		deps.Env.VERSION,
	)
	core := targon.CreateCore(deps)

	numbPtr := flag.Int("uid", -1, "Miner uid")
	flag.Parse()
	if *numbPtr == -1 {
		core.Deps.Log.Fatal("No UID selected, exiting")
	}

	blockHash, err := core.Deps.Client.Api.RPC.Chain.GetBlockHashLatest()
	if err != nil {
		core.Deps.Log.Errorw("Failed getting blockhash for neurons", "error", err)
		return
	}
	neuron, err := runtime.GetNeuron(core.Deps.Client, uint16(core.Deps.Env.NETUID), uint16(*numbPtr), &blockHash)
	if err != nil {
		core.Deps.Log.Errorw("Failed getting neurons", "error", err)
		return
	}
	core.Deps.Log.Infof("Axon Info: %+v", neuron)

	tr := &http.Transport{
		TLSHandshakeTimeout: 5 * time.Second,
		MaxConnsPerHost:     1,
		DisableKeepAlives:   true,
	}
	client := &http.Client{Transport: tr, Timeout: 5 * time.Minute}

	nodes, err := targon.GetCVMNodes(core, client, neuron)
	if err != nil {
		core.Deps.Log.Fatalf("Failed to get nodes: %s", err)
	}
	for _, n := range nodes {
		gpus, _, err := targon.CheckCVMAttest(core, client, neuron, n)
		if err != nil {
			core.Deps.Log.Infow("CVM attest error", "err", err)
			continue
		}
		core.Deps.Log.Infow("CVM attest results", "node", n, "gpus", fmt.Sprintf("%v", gpus))
	}
}
