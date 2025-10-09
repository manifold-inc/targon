package callbacks

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"sort"
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
	client := &http.Client{Transport: tr, Timeout: 5 * time.Second}
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

	// auction -> bid -> total gpus
	bidcounts := map[string]map[int]int{}
	for a := range c.Auctions {
		bidcounts[a] = map[int]int{}
	}

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
				continue
			}

			// ensure price is between 1 and max bid
			// price is cents per hour per gpu
			price := max(min(n.Price, auc.MaxBid), 1)

			bidCount := 1
			if auc.MinClusterSize != 0 {
				bidCount = len(*c.VerifiedNodes[uid][n.IP].GPUCards)
			}

			auction[auctionName] = append(auction[auctionName], &targon.MinerBid{
				IP:    n.IP,
				Price: price,
				UID:   uid,
				Count: bidCount,
			})

			// This is used to calcualte each ring in the next steps
			bidcounts[auctionName][price] += bidCount
		}
	}

	// uid -> % of emission
	payouts := map[string]float64{}

	// auction -> bid count (i.e gpus are counted per gpu in a node)
	paidnodes := map[string]int{}

	// For each auction, sort the bids in ascending order and accept bids untill
	// we have hit the cap for this auction
	for auctiontype, pool := range c.Auctions {
		lastPrice := 0
		tiedPayouts := map[string]int{}
		sort.Slice(auction[auctiontype], func(i, j int) bool {
			return auction[auctiontype][i].Price < auction[auctiontype][j].Price
		})
		// max % of the pool for this auction
		maxEmission := float64(pool.Emission) / 100
		emissionSum := 0.0
		tiedBids := 0
		isRingOverMax := false
		for _, bid := range auction[auctiontype] {
			// GPUs * the bid price/h * interval duration approx / emission pool
			// 	== percent of emission pool for this node for this interval
			thisEmission := (float64(bid.Count) * ((float64(bid.Price) / 100) * 1.2)) / *c.EmissionPool
			paidnodes[auctiontype] += bid.Count

			// On each bid increase, check if ring is over max
			if bid.Price != lastPrice && !isRingOverMax {
				isRingOverMax = ((thisEmission/float64(bid.Count))*float64(bidcounts[auctiontype][bid.Price]))+emissionSum > maxEmission
			}
			if isRingOverMax {
				tiedPayouts[bid.UID] += bid.Count
				tiedBids += bid.Count
				bid.Diluted = true
				continue
			}
			bid.Diluted = false
			emissionSum += thisEmission
			payouts[bid.UID] += thisEmission
			lastPrice = bid.Price
			bid.Payout = (float64(bid.Price) / 100.0) * float64(bid.Count)
		}

		// Just skip if there is not much emission left to split
		if maxEmission-emissionSum < .01 {
			continue
		}
		// If not all rings get paid their bids, normalize all other rings to the remaning emission
		// and add that to the payouts. This greatly increases downward price pressure
		// by highly rewarding people that underbid the last paid ring if it ties.
		maxTiedEmissionBidPool := (float64(tiedBids) * (float64(pool.MaxBid) / 100)) / *c.EmissionPool
		remainingEmission := maxEmission - emissionSum
		dilutedPayoutPerGPU := (min(remainingEmission, maxTiedEmissionBidPool) * *c.EmissionPool) / float64(tiedBids)
		for _, bid := range auction[auctiontype] {
			if bid.Diluted {
				bid.Payout = (dilutedPayoutPerGPU * float64(bid.Count)) / 1.2
				payouts[bid.UID] += (dilutedPayoutPerGPU * float64(bid.Count)) / *c.EmissionPool
			}
		}
	}

	var finalScores []uint16
	var finalUids []uint16
	sumScores := uint16(0)
	for uid, payout := range payouts {
		fw := math.Floor(float64(setup.U16MAX) * payout)
		if fw == 0 {
			continue
		}
		thisScore := uint16(fw)
		uidInt, _ := strconv.Atoi(uid)
		finalScores = append(finalScores, thisScore)
		finalUids = append(finalUids, uint16(uidInt))
		sumScores += thisScore
	}

	//  final burn keys
	burnKeys := []int{28}

	// For each burn key, send a random amount
	forBurn := setup.U16MAX - sumScores
	for _, k := range burnKeys {
		if forBurn == 0 {
			continue
		}
		thisBurn := rand.Intn(int(forBurn))
		forBurn -= uint16(thisBurn)

		finalScores = append(finalScores, uint16(thisBurn))
		finalUids = append(finalUids, uint16(k))
	}
	// Add remaning burn to last burn key
	// If there was no burn this is a noop
	finalScores[len(finalScores)-1] += forBurn

	c.Deps.Log.Infow("Payouts", "percentages", fmt.Sprintf("%+v", payouts), "gpus", fmt.Sprintf("%+v", paidnodes))
	c.Deps.Log.Infow("Miner scores", "uids", fmt.Sprintf("%v", finalUids), "scores", fmt.Sprintf("%v", finalScores))

	return finalUids, finalScores, auction, nil
}
