package callbacks

import (
	"errors"
	"fmt"
	"math"
	"sort"
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

type MinerBid struct {
	targon.MinerNode
	uid  string
	gpus int
}

func getWeights(c *targon.Core) ([]types.U16, []types.U16, error) {
	if c.EmissionPool == nil {
		return []types.U16{}, []types.U16{}, errors.New("emission pool is not set")
	}

	// auction => miner nodes
	auction := map[string][]MinerBid{}

	bidcounts := map[string]map[int]int{}

	// For each uid, for each node, add any passing nodes to the auction map
	// under the respective auction
	for uid, nodes := range c.MinerNodes {
		for _, n := range nodes {
			if c.PassedAttestation[uid] == nil {
				continue
			}
			if c.PassedAttestation[uid][n.Ip] == nil {
				continue
			}
			if len(c.PassedAttestation[uid][n.Ip]) == 0 {
				continue
			}
			gpu := strings.ToLower(c.PassedAttestation[uid][n.Ip][0])
			for auctionBucket := range c.Auctions {
				if _, ok := bidcounts[auctionBucket]; !ok {
					bidcounts[auctionBucket] = map[int]int{}
				}
				if strings.Contains(gpu, auctionBucket) {
					auction[auctionBucket] = append(auction[auctionBucket], MinerBid{
						MinerNode: *n,
						uid:       uid,
						gpus:      len(c.PassedAttestation[uid][n.Ip]),
					})

					// NOTE::TODO
					// This assumes all prices for nodes are nodes of the same cluster
					// size, might need to rework for actual release.
					// Best method might be to calculate this rungs price sum right here,
					// including gpu count. TBD
					bidcounts[auctionBucket][n.Price] += 1
					break
				}
			}
		}
	}

	// uid -> % of emission
	payouts := map[string]float64{}

	paidnodes := map[string]int{}

	// For each auction, sort the bids in ascending order and accept bids untill
	// we have hit the cap for this auction
	lastPrice := 0
	for auctiontype, pool := range c.Auctions {
		tiedPayouts := map[string]float64{}
		sort.Slice(auction[auctiontype], func(i, j int) bool {
			return auction[auctiontype][i].Price < auction[auctiontype][j].Price
		})
		c.Deps.Log.Debugf("Sorted Auction entries: %v", auction[auctiontype])
		// max % of the pool for this auction
		maxEmission := float64(pool) / 100
		emissionSum := 0.0
		tiedSum := 0.0
		isRingOverMax := false
		for _, bid := range auction[auctiontype] {
			// GPUs * the bid price/h * interval duration approx / emission pool
			// 	== percent of emission pool for this node for this interval
			thisEmission := (float64(bid.gpus) * ((float64(bid.Price) / 100) * 1.233)) / *c.EmissionPool
			if thisEmission+emissionSum > maxEmission && bid.Price != lastPrice {
				c.Deps.Log.Infof("Auction ending at ring: %d", bid.Price)
				break
			}
			paidnodes[auctiontype] += bid.gpus
			if bid.Price != lastPrice {
				isRingOverMax = (thisEmission*float64(bidcounts[auctiontype][bid.Price]))+emissionSum > maxEmission
			}
			if isRingOverMax {
				c.Deps.Log.Infof("UID %s bid diluted in last ring: %d", bid.uid, bid.Price)
				tiedPayouts[bid.uid] += thisEmission
				tiedSum += thisEmission
				continue
			}
			c.Deps.Log.Infof("UID %s bid fully included: %d", bid.uid, bid.Price)
			emissionSum += thisEmission
			payouts[bid.uid] += thisEmission
			lastPrice = bid.Price
		}

		// If the last ring ties, normalize that ring to the remaning emission
		// and add that to the payouts. This greatly increases downward price pressure
		// by highly rewarding people that underbid the last paid ring if it ties.
		for uid, payout := range tiedPayouts {
			diluted := (payout / tiedSum) * (maxEmission - emissionSum)
			c.Deps.Log.Infof("UID %s diluted to: %.2f%%", uid, diluted*100)
			payouts[uid] += diluted
		}

	}

	var finalScores []types.U16
	var finalUids []types.U16
	sumScores := uint16(0)
	for uid, payout := range payouts {
		fw := math.Floor(float64(setup.U16MAX) * payout)
		if fw == 0 {
			continue
		}
		thisScore := uint16(fw)
		uidInt, _ := strconv.Atoi(uid)
		finalScores = append(finalScores, types.NewU16(thisScore))
		finalUids = append(finalUids, types.NewU16(uint16(uidInt)))
		sumScores += thisScore
	}
	burnKey := 28
	finalScores = append(finalScores, types.NewU16(setup.U16MAX-sumScores))
	finalUids = append(finalUids, types.NewU16(uint16(burnKey)))

	c.Deps.Log.Infow("Payouts", "percentages", fmt.Sprintf("%+v", payouts), "gpus", fmt.Sprintf("%+v", paidnodes))
	c.Deps.Log.Infow("Miner scores", "uids", fmt.Sprintf("%v", finalUids), "scores", fmt.Sprintf("%v", finalScores))

	return finalUids, finalScores, nil
}
