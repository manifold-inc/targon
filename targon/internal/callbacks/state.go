package callbacks

import (
	"targon/internal/targon"
	"targon/internal/tower"

	"github.com/subtrahend-labs/gobt/runtime"
)

func resetState(c *targon.Core) {
	c.Neurons = make(map[string]runtime.NeuronInfo)
	c.MinerNodes = make(map[string][]*targon.MinerNode)
	c.NodeIds = make(map[string]bool)
	// Dont really need to wipe tao price
	c.EmissionPool = nil
	c.MinerErrors = make(map[string]map[string]string)
	c.VerifiedNodes = make(map[string]map[string]*targon.UserData)
	c.Auctions = make(map[string]tower.Auction)
	c.AuctionResults = make(map[string][]*targon.MinerBid)
	c.TaoPrice = nil
}
