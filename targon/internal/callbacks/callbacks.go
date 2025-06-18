package callbacks

import (
	"math/rand"
	"time"

	"targon/internal/pyth"
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
		go logBlockCallback(c, h)
	})

	// get neurons
	v.AddBlockCallback(func(h types.Header) {
		// Run after first block of interval
		if h.Number%360 != 1 && len(c.Neurons) != 0 {
			return
		}
		getNeuronsCallback(v, c, h)
	})

	// get emission for this interval
	v.AddBlockCallback(func(h types.Header) {
		if c.EmissionPool != nil {
			return
		}
		taoPrice, err := pyth.GetTaoPrice()
		if err != nil {
			c.Deps.Log.Errorw("Failed getting tao price", "error", err)
			return
		}
		c.TaoPrice = &taoPrice
		c.Deps.Log.Infof("Current tao price $%f", *c.TaoPrice)
		p, err := storage.GetSubnetTaoInEmission(c.Deps.Client, types.NewU16(uint16(c.Deps.Env.NETUID)), &h.ParentHash)
		if err != nil {
			c.Deps.Log.Errorw("Failed getting sn tao emissions", "error", err)
			return
		}
		emi := (float64(*p) / 1e9) * .41 * 360 * *c.TaoPrice
		c.EmissionPool = &emi
		c.Deps.Log.Infof("Current sn miner emission pool in $ %f", *c.EmissionPool)
	})

	// get miner nodes
	v.AddBlockCallback(func(h types.Header) {
		if h.Number%360 != 1 && len(c.MinerNodes) != 0 {
			return
		}
		getMinerNodes(c)
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
		chance := rand.Float32()
		if chance < .95 && len(c.PassedAttestation) != 0 {
			return
		}
		getPassingAttestations(c)
	})

	// Ping miner healthchecks
	v.AddBlockCallback(func(h types.Header) {
		if h.Number%10 != 0 || len(c.MinerNodes) == 0 {
			return
		}
		pingHealthChecks(c)
	})

	// Log weights
	v.AddBlockCallback(func(h types.Header) {
		if h.Number%10 != 0 || len(c.MinerNodes) == 0 {
			return
		}
		logWeights(c)
	})

	// Discord Notifications
	v.AddBlockCallback(func(h types.Header) {
		if h.Number%720 != 0 {
			return
		}
		sendDailyGPUSummary(c, h)
	})

	// Set Weights
	v.AddBlockCallback(func(h types.Header) {
		if h.Number%360 != 0 || len(c.MinerNodes) == 0 {
			return
		}
		if c.Deps.Mongo != nil {
			err := targon.SyncMongo(c, int(h.Number))
			if err != nil {
				c.Deps.Log.Errorw("failed syncing to mongo", "error", err)
			}
		}
		setWeights(v, c, h)
	})
}
