package callbacks

import (
	"math/rand"
	"time"

	"targon/internal/targon"

	"github.com/centrifuge/go-substrate-rpc-client/v4/types"
	"github.com/subtrahend-labs/gobt/boilerplate"
	"github.com/subtrahend-labs/gobt/storage"
)

// TODO
// Confrim set weight hash success
func AddBlockCallbacks(v *boilerplate.BaseChainSubscriber, c *targon.Core) {
	// block timer for catching hangs
	t := time.AfterFunc(1*time.Hour, func() {
		c.Deps.Log.Error("havint seen any blocks in over an hour, am i stuck?")
	})
	v.AddBlockCallback(func(h types.Header) {
		t.Reset(1 * time.Hour)
	})

	// Logging blocks
	v.AddBlockCallback(func(h types.Header) {
		logBlockCallback(c, h)
	})

	// get neurons
	v.AddBlockCallback(func(h types.Header) {
		// Run after first block of interval
		if h.Number%360 != 1 && len(c.Neurons) != 0 {
			return
		}
		getNeuronsCallback(c, h)
	})

	// get emission and auction data for this interval
	v.AddBlockCallback(func(h types.Header) {
		if c.EmissionPool != nil {
			return
		}
		// Get tower pyth price and emission slice along with min burn
		auctionData, err := c.Deps.Tower.AuctionDetails()
		if err != nil {
			c.Deps.Log.Errorw("Failed getting tao price", "error", err)
			return
		}
		c.TaoPrice = &auctionData.TaoPrice
		c.MaxBid = auctionData.MaxBid
		c.Auctions = auctionData.Auctions
		c.Deps.Log.Infof("Auctions: %+v", c.Auctions)
		c.Deps.Log.Infof("Max Bid: %d", c.MaxBid)
		c.Deps.Log.Infof("Current tao price $%f", *c.TaoPrice)
		p, err := storage.GetSubnetTaoInEmission(c.Deps.Client, types.NewU16(uint16(c.Deps.Env.NETUID)), &h.ParentHash)
		if err != nil {
			c.Deps.Log.Errorw("Validator is falling behind current block time")
			p, err = storage.GetSubnetTaoInEmission(c.Deps.Client, types.NewU16(uint16(c.Deps.Env.NETUID)), nil)
			if err != nil {
				c.Deps.Log.Errorw("Failed getting sn tao emissions", "error", err)
				return
			}
		}
		emi := (float64(*p) / 1e9) * .41 * 360 * *c.TaoPrice
		c.EmissionPool = &emi
		c.Deps.Log.Infof("Current sn miner emission pool in $ %f", *c.EmissionPool)
	})

	// get miner nodes
	// Every 30 blocks off the internval tempo untill 180 left in block
	v.AddBlockCallback(func(h types.Header) {
		if (h.Number%30 != 1 || h.Number%360 > 180) && len(c.MinerNodes) != 0 {
			return
		}
		getNodesAll(c)
	})

	// get passing attestations
	v.AddBlockCallback(func(h types.Header) {
		if c.Neurons == nil {
			return
		}
		blocksTill := 360 - (h.Number % 360)
		if blocksTill < 20 {
			return
		}
		// Not on specific tempo;
		// helps reduce stress on cvm nodes from number of pings
		chance := rand.Float64()
		if chance < c.Deps.Env.ATTEST_RATE && len(c.PassedAttestation) != 0 {
			return
		}
		getPassingAttestations(c)
	})

	// Log weights
	v.AddBlockCallback(func(h types.Header) {
		if h.Number%10 != 0 || len(c.MinerNodes) == 0 {
			return
		}
		logWeights(c)
	})

	// Set Weights
	v.AddBlockCallback(func(h types.Header) {
		if h.Number%360 != 0 || len(c.MinerNodes) == 0 {
			return
		}
		uids, scores, results, err := getWeights(c)
		if err != nil {
			c.Deps.Log.Errorw("Failed getting weights", "error", err)
			return
		}
		c.AuctionResults = results

		err = sendIntervalSummary(c, h, uids, scores)
		if err != nil {
			c.Deps.Log.Warnw("Failed logging to discord", "error", err)
		}
		if c.Deps.Mongo != nil {
			syncErr := targon.SyncMongo(c, uids, scores, h)
			if syncErr != nil {
				c.Deps.Log.Errorw("Failed syncing complete data to mongo", "error", syncErr)
			}

			if syncErr == nil {
				c.Deps.Log.Infow("Weights set and complete data synced",
					"uids", uids,
					"scores", scores,
					"block", h.Number,
					"miners", len(uids))
			}
		}
		setWeights(c, uids, scores)

		// catches up to live block time
		v.Restart()
	})
}
