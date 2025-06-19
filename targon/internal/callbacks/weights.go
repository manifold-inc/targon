package callbacks

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"

	"targon/internal/setup"
	"targon/internal/targon"

	"github.com/centrifuge/go-substrate-rpc-client/v4/types"
	"github.com/subtrahend-labs/gobt/boilerplate"
	"github.com/subtrahend-labs/gobt/extrinsics"
	"github.com/subtrahend-labs/gobt/sigtools"
)

func setWeights(v *boilerplate.BaseChainSubscriber, c *targon.Core, uids []types.U16, scores []types.U16) {
	c.Mu.Lock()
	defer func() {
		c.Mu.Unlock()
		resetState(c)
	}()

	c.Deps.Log.Infow(
		"Setting Weights",
		"uids",
		fmt.Sprintf("%+v", uids),
		"scores",
		fmt.Sprintf("%+v", scores),
	)

	if c.Deps.Env.DEBUG {
		c.Deps.Log.Warn("Skipping weightset due to debug flag")
		return
	}
	// Actually set weights
	ext, err := extrinsics.SetWeightsExt(
		c.Deps.Client,
		types.U16(v.NetUID),
		uids,
		scores,
		c.Deps.Env.VERSION,
	)
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

func getWeights(c *targon.Core) ([]types.U16, []types.U16, error) {
	// TODO some sort of multi-check per interval
	if c.EmissionPool == nil {
		return []types.U16{}, []types.U16{}, errors.New("emission pool is not set")
	}
	minerCut := 0.0
	var uids []types.U16
	var scores []float64
	var cvmNodes []string
	gpus := map[string]int{}
	// for each uid
	for uid, nodes := range c.MinerNodes {
		thisScore := 0.0
		// for each node
		for _, n := range nodes {
			if c.PassedAttestation[uid] == nil {
				continue
			}
			if c.PassedAttestation[uid][n] == nil {
				continue
			}
			cvmNodes = append(cvmNodes, n)
			// for each gpu
			for _, gpu := range c.PassedAttestation[uid][n] {
				ml := strings.ToLower(gpu)
				gpus[ml] += 1
				// score is GPU cost per hour * hours in interval (1.5) / total coming
				// into sn this interval
				switch {
				case strings.Contains(ml, "h100"):
					score := (2.5 * 1.233) / *c.EmissionPool
					thisScore += score
					minerCut += score
				case strings.Contains(ml, "h200"):
					score := (3.5 * 1.233) / *c.EmissionPool
					thisScore += score
					minerCut += score
				default:
					continue
				}
			}
		}
		if thisScore < 0.01 {
			continue
		}
		uidInt, _ := strconv.Atoi(uid)

		uids = append(uids, types.NewU16(uint16(uidInt)))
		scores = append(scores, thisScore)
	}
	burnKey := 28
	minerCut = math.Min(minerCut, 1) // Diulte after 100% emission hit
	scores = normalize(scores, minerCut)
	scores = append(scores, 1-minerCut)
	uids = append(uids, types.NewU16(uint16(burnKey)))

	for gpu, count := range gpus {
		c.Deps.Log.Infof("%s count: %d", gpu, count)
	}
	c.Deps.Log.Infof("CVM IPs: %v", cvmNodes)
	c.Deps.Log.Infow("Miner scores", "uids", fmt.Sprintf("%v", uids), "scores", fmt.Sprintf("%v", scores))

	var finalScores []types.U16
	var finalUids []types.U16
	sumScores := uint16(0)
	for i, s := range scores {
		// send dust to burn
		if i == len(scores)-1 {
			continue
		}
		fw := math.Floor(float64(setup.U16MAX) * s)
		if fw == 0 {
			continue
		}
		thisScore := uint16(fw)
		finalScores = append(finalScores, types.NewU16(thisScore))
		finalUids = append(finalUids, uids[i])
		sumScores += thisScore
	}
	finalScores = append(finalScores, types.NewU16(setup.U16MAX-sumScores))
	finalUids = append(finalUids, types.NewU16(uint16(burnKey)))

	return finalUids, finalScores, nil
}

func normalize(arr []float64, sumTo float64) []float64 {
	sum := 0.0
	for _, num := range arr {
		sum += num
	}
	if sum == 0.0 {
		return arr
	}
	newArr := []float64{}
	for _, num := range arr {
		newArr = append(newArr, (num/sum)*sumTo)
	}
	return newArr
}
