package targon

import (
	"fmt"
	"net/http"
	"sync"
	"time"

	"targon/validator"

	"github.com/centrifuge/go-substrate-rpc-client/v4/types"
	"github.com/subtrahend-labs/gobt/runtime"
)

func AddBlockCallbakcs(v *validator.BaseValidator, c *Core) {
	v.AddBlockCallback(func(h types.Header) {
		logBlockCallback(c, h)
	})
	v.AddBlockCallback(func(h types.Header) {
		resetState(c, h)
	})
	v.AddBlockCallback(func(h types.Header) {
		getNeuronsCallback(v, c, h)
	})
	v.AddBlockCallback(func(h types.Header) {
		getCVMNodesCallback(c, h)
	})
}

func logBlockCallback(c *Core, h types.Header) {
	// Run Every Block
	c.Deps.Log.Infow("New block", "block", fmt.Sprintf("%v", h.Number), "left_in_interval", fmt.Sprintf("%d", 360-(h.Number%360)))
}

func getNeuronsCallback(v *validator.BaseValidator, c *Core, h types.Header) {
	// Run after first block of interval
	if h.Number%360 != 1 && c.Neurons != nil {
		return
	}
	c.Deps.Log.Info("Updating neurons")
	blockHash, err := c.Deps.Client.Api.RPC.Chain.GetBlockHash(uint64(h.Number))
	if err != nil {
		c.Deps.Log.Errorw("Failed getting blockhash for neurons", "error", err)
		return
	}
	neurons, err := runtime.GetNeurons(c.Deps.Client, uint16(v.NetUID), &blockHash)
	if err != nil {
		c.Deps.Log.Errorw("Failed getting neurons", "error", err)
		return
	}
	c.NeuronsMu.Lock()
	defer c.NeuronsMu.Unlock()
	c.Neurons = neurons
	c.Deps.Log.Info("Neurons Updated")
}

func resetState(c *Core, h types.Header) {
	if h.Number%360 != 1 && c.NeuronNodes != nil {
		return
	}
	c.NeuronNodes = map[string][]string{}
}

func getCVMNodesCallback(c *Core, h types.Header) {
	if h.Number%10 != 1 || c.Neurons == nil {
		return
	}
	tr := &http.Transport{
		ResponseHeaderTimeout: 2 * time.Second,
		MaxConnsPerHost:       1,
		DisableKeepAlives:     true,
	}
	client := &http.Client{Transport: tr, Timeout: 5 * time.Second}
	wg := sync.WaitGroup{}
	wg.Add(len(c.Neurons))
	for _, n := range c.Neurons {
		go func() {
			GetCVMNodes(c, client, &n)
			wg.Done()
		}()
	}
	wg.Wait()
	c.Deps.Log.Infof("Found %d miners with nodes", len(c.NeuronNodes))
	for k, v := range c.NeuronNodes {
		c.Deps.Log.Infow("nodes for "+k, "nodes", len(v))
	}
}
