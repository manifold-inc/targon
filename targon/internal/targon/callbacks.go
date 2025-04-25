package targon

import (
	"fmt"
	"math"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"targon/internal/setup"

	"github.com/centrifuge/go-substrate-rpc-client/v4/types"
	"github.com/subtrahend-labs/gobt/boilerplate"
	"github.com/subtrahend-labs/gobt/extrinsics"
	"github.com/subtrahend-labs/gobt/runtime"
	"github.com/subtrahend-labs/gobt/sigtools"
)

func AddBlockCallbakcs(v *boilerplate.BaseChainSubscriber, c *Core) {
	v.AddBlockCallback(func(h types.Header) {
		go logBlockCallback(c, h)
	})
	v.AddBlockCallback(func(h types.Header) {
		getNeuronsCallback(v, c, h)
	})
	v.AddBlockCallback(func(h types.Header) {
		getCVMNodesCallback(c, h)
	})
	v.AddBlockCallback(func(h types.Header) {
		logWeights(c, h)
	})
	if !c.Deps.Env.DEBUG {
		v.AddBlockCallback(func(h types.Header) {
			setWeights(v, c, h)
		})
	}
}

func logBlockCallback(c *Core, h types.Header) {
	// Run Every Block
	c.Deps.Log.Infow("New block", "block", fmt.Sprintf("%v", h.Number), "left_in_interval", fmt.Sprintf("%d", 360-(h.Number%360)))
}

func getNeuronsCallback(v *boilerplate.BaseChainSubscriber, c *Core, h types.Header) {
	// Run after first block of interval
	if h.Number%360 != 1 && c.Neurons != nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
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
	c.Neurons = neurons
	c.Deps.Log.Info("Neurons Updated")
}

func resetState(c *Core) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.NeuronHardware = map[string][]string{}
}

func getCVMNodesCallback(c *Core, h types.Header) {
	if h.Number%10 != 1 || c.Neurons == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	tr := &http.Transport{
		TLSHandshakeTimeout:   5 * time.Second,
		MaxConnsPerHost:       1,
		DisableKeepAlives:     true,
	}
	client := &http.Client{Transport: tr, Timeout: 1 * time.Minute}
	wg := sync.WaitGroup{}
	wg.Add(len(c.Neurons))
	c.Deps.Log.Infof("Checking CVM nodes for %d miners", len(c.Neurons))
	for _, n := range c.Neurons {
		go func() {
			defer wg.Done()
			GetCVMNodes(c, client, &n)
		}()
	}
	wg.Wait()
	c.Deps.Log.Infof("Found %d miners with nodes", len(c.NeuronHardware))

	var beersData []GPUData
	for k, v := range c.NeuronHardware {
		beersData = append(beersData, GPUData{Uid: k, Data: v})
		c.Deps.Log.Infow("nodes for "+k, "nodes", len(v))
	}
	if err := sendGPUDataToBeers(c, client, beersData); err != nil {
		c.Deps.Log.Warnw("Failed to send GPU data to beers", "error", err)
	}
}

func logWeights(c *Core, h types.Header) {
	if h.Number%15 != 2 || c.NeuronHardware == nil {
		return
	}
	uids, scores := getWeights(c)
	c.Deps.Log.Infow("Current Weights", "uids", fmt.Sprintf("%+v", uids), "scores", fmt.Sprintf("%+v", scores))
}

func setWeights(v *boilerplate.BaseChainSubscriber, c *Core, h types.Header) {
	if h.Number%360 != 0 || c.NeuronHardware == nil {
		return
	}
	c.mu.Lock()
	defer func() {
		c.mu.Unlock()
		resetState(c)
	}()
	uids, scores := getWeights(c)
	c.Deps.Log.Info("Setting Weights", "uids", fmt.Sprintf("%+v", uids), "scores", fmt.Sprintf("%+v", scores))
	// Actually set weights
	ext, err := extrinsics.SetWeightsExt(c.Deps.Client, types.U16(v.NetUID), uids, scores, c.Deps.Env.VERSION)
	if err != nil {
		c.Deps.Log.Warnw("Failed creating setweights ext", "error", err)
		return
	}
	ops, err := sigtools.CreateSigningOptions(c.Deps.Client, c.Deps.Hotkey, nil)
	if err != nil {
		c.Deps.Log.Errorw("Failed creating sigining opts", "error", err)
		return
	}
	err = ext.Sign(
		c.Deps.Hotkey,
		c.Deps.Client.Meta,
		ops...,
	)
	if err != nil {
		c.Deps.Log.Errorw("Error signing setweights", "error", err)
		return
	}

	hash, err := c.Deps.Client.Api.RPC.Author.SubmitExtrinsic(*ext)
	if err != nil {
		c.Deps.Log.Errorw("Error submitting extrinsic", "error", err)
		return
	}
	c.Deps.Log.Infow("Set weights on chain successfully", "hash", hash.Hex())
}

func getWeights(c *Core) ([]types.U16, []types.U16) {
	// TODO some sort of multi-check per interval
	var uids []types.U16
	var scores []float64
	for k, v := range c.NeuronHardware {
		thisScore := 0.0
		for _, m := range v {
			ml := strings.ToLower(m)
			switch {
			case strings.Contains(ml, "h100"):
				thisScore += 1
			case strings.Contains(ml, "h200"):
				thisScore += 2
			default:
				continue
			}
		}
		if thisScore < 0.01 {
			continue
		}
		uidInt, _ := strconv.Atoi(k)

		uids = append(uids, types.NewU16(uint16(uidInt)))
		scores = append(scores, thisScore)
	}
	minerCut := .15
	burnKey := 28
	scores = Normalize(scores, minerCut)
	scores = append(scores, 1-minerCut)
	uids = append(uids, types.NewU16(uint16(burnKey)))

	var finalScores []types.U16
	var finalUids []types.U16
	for i, s := range scores {
		fw := math.Round(float64(setup.U16MAX) * s)
		if fw == 0 {
			continue
		}
		finalScores = append(finalScores, types.NewU16(uint16(fw)))
		finalUids = append(finalUids, uids[i])
	}

	return finalUids, finalScores
}
