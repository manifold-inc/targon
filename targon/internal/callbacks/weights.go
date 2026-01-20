package callbacks

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"strconv"
	"time"

	"targon/internal/setup"
	"targon/internal/targon"
)

type WeightsAPIBody struct {
	Uids    []uint16 `json:"uids"`
	Weights []uint16 `json:"weights"`
	Version int      `json:"version"`
}

// final burn keys, randomized across 3 keys for WC combat
var burnKeys = []int{28, 15, 243}

func setWeights(c *targon.Core, uids []uint16, scores []uint16) {
	defer func() {
		resetState(c)
	}()

	c.Deps.Log.Infow(
		"Setting Weights",
		"uids",
		fmt.Sprintf("%+v", uids),
		"scores",
		fmt.Sprintf("%+v", scores),
	)

	if c.Deps.Env.Debug {
		c.Deps.Log.Warn("Skipping weightset due to debug flag")
		return
	}
	tr := &http.Transport{
		TLSHandshakeTimeout: 5 * time.Second,
		MaxConnsPerHost:     1,
		DisableKeepAlives:   true,
	}
	// Needs to wait till it finishes setting weights on chain
	client := &http.Client{Transport: tr, Timeout: 2 * time.Minute}
	body, err := json.Marshal(WeightsAPIBody{Uids: uids, Weights: scores, Version: int(c.Deps.Env.Version)})
	if err != nil {
		c.Deps.Log.Errorw("failed marshaling weights", "error", err)
		return
	}
	req, err := http.NewRequest("POST", "http://weights-api/api/v1/set-weights", bytes.NewBuffer(body))
	if err != nil {
		c.Deps.Log.Errorw("failed generating set weights req", "error", err)
		return
	}
	res, err := client.Do(req)
	if err != nil {
		c.Deps.Log.Errorw("failed setting weights", "error", err)
		return
	}
	b, err := io.ReadAll(res.Body)
	if err != nil {
		c.Deps.Log.Errorw("failed reading response from set weights", "error", err)
		return
	}
	if res.StatusCode != 200 {
		c.Deps.Log.Errorw("set-weights failed", "error", string(b))
		return
	}
	var msg *SetWeightsRes
	err = json.Unmarshal(b, &msg)
	if err != nil {
		c.Deps.Log.Errorw("failed setting weights", "error", string(b))
		return
	}
	if !msg.Success {
		c.Deps.Log.Errorw("failed setting weights", "error", msg.Msg)
		return
	}
	c.Deps.Log.Infow("Set weights on chain successfully", "msg", msg.Msg)
}

type SetWeightsRes struct {
	Success bool   `json:"success,omitempty"`
	Msg     string `json:"msg,omitempty"`
}

// getWeights returns
// uid array, weights array, auction results, error
func getWeights(c *targon.Core) ([]uint16, []uint16, map[string][]*targon.MinerBid, error) {
	if c.EmissionPool == nil {
		return nil, nil, nil, errors.New("emission pool is not set")
	}

	// auction => miner nodes
	auction := map[string][]*targon.MinerBid{}

	// For each uid, for each node, add any passing nodes to the auction map
	// under the respective auction

	for uid, nodes := range c.MinerNodes {
		if c.VerifiedNodes[uid] == nil {
			continue
		}
		for _, n := range nodes {
			if c.VerifiedNodes[uid][n.IP] == nil {
				continue
			}
			auctionName := c.VerifiedNodes[uid][n.IP].AuctionName
			auc, ok := c.Auctions[auctionName]
			if !ok {
				continue
			}

			// Node does not have enough GPUS. cpus will have zero min cluser size
			// forces certian number of gpus per node
			if auc.MinClusterSize != 0 && auc.MinClusterSize > len(*c.VerifiedNodes[uid][n.IP].GPUCards) {
				c.MinerErrors[uid][n.IP] = fmt.Sprintf("Node reporting %d cards, %d required", len(*c.VerifiedNodes[uid][n.IP].GPUCards), auc.MinClusterSize)
				continue
			}

			cards := 1
			if auc.MinClusterSize != 0 {
				cards = len(*c.VerifiedNodes[uid][n.IP].GPUCards)
			}

			auction[auctionName] = append(auction[auctionName], &targon.MinerBid{
				IP:    n.IP,
				UID:   uid,
				Count: cards,
			})
		}
	}

	// uid -> % of emission
	payouts := map[string]float64{}

	// For each auction pay up to max per node,
	// and target price for target gpus
	for auctiontype, aucInfo := range c.Auctions {
		pool := aucInfo.TargetPrice * aucInfo.TargetNodes
		var nodes int
		for _, bid := range auction[auctiontype] {
			nodes += bid.Count
		}
		if nodes == 0 {
			continue
		}
		perMiner := min(pool/nodes, aucInfo.MaxPrice)

		for _, bid := range auction[auctiontype] {
			// Miner incentive is the % of emission pool they should get
			minerIncentive := (float64(perMiner) * 1.2 * float64(bid.Count)) / (*c.EmissionPool * 100)
			payouts[bid.UID] += minerIncentive
			bid.Payout = (float64(perMiner) * float64(bid.Count)) / 100
		}
	}

	var finalScores []uint16
	var finalUids []uint16
	sumScores := 0
	// NOTE we dont actually need to normalize here in the case that
	// our pool sum > emission pool. This is because we are sending
	// weights via the bittensor sdk which is going to normalize anyways,
	// so no use doing it here.
	for uid, payout := range payouts {
		fw := int(math.Floor(float64(setup.U16MAX) * payout))
		if fw == 0 {
			continue
		}
		sumScores += fw
		thisScore := uint16(fw)
		uidInt, _ := strconv.Atoi(uid)
		finalScores = append(finalScores, thisScore)
		finalUids = append(finalUids, uint16(uidInt))
	}

	// This is the only part that needs taken care of if sum of pools is greater
	// than emission pool
	forBurn := max(setup.U16MAX-sumScores, 0)

	// For each burn key, send a random amount
	for uid, percent := range c.BurnDistribution {
		if forBurn == 0 {
			continue
		}
		// This is kinda cursed, i know
		thisBurn := uint16(math.Floor(float64(forBurn) * (float64(percent) / 100)))
		forBurn -= int(thisBurn)
		// Just to be safe from int underflow on u16, forburn is an int that we make sure never goes below 0
		forBurn = int(max(forBurn, 0))

		finalScores = append(finalScores, uint16(thisBurn))
		finalUids = append(finalUids, uint16(uid))
	}
	// Add remaning burn to last burn key
	// If there was no burn this is a noop
	finalScores[len(finalScores)-1] += uint16(forBurn)

	c.Deps.Log.Infow("Payouts", "percentages", fmt.Sprintf("%+v", payouts))
	c.Deps.Log.Infow("Miner scores", "uids", fmt.Sprintf("%v", finalUids), "scores", fmt.Sprintf("%v", finalScores))

	return finalUids, finalScores, auction, nil
}
