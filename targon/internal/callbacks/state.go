package callbacks

import (
	"targon/internal/targon"

	"github.com/subtrahend-labs/gobt/runtime"
)

func resetState(c *targon.Core) {
	c.Mu.Lock()
	defer c.Mu.Unlock()
	c.Neurons = make(map[string]runtime.NeuronInfo)
	c.MinerNodes = make(map[string][]*targon.MinerNode)
	c.GPUids = make(map[string]bool)
	// Dont really need to wipe tao price
	c.EmissionPool = nil
	c.AttestErrors = make(map[string]map[string]string)
	c.PassedAttestation = make(map[string]map[string][]string)
	c.Auctions = make(map[string]int)
	c.AuctionResults = make(map[string][]*targon.MinerBid)
	c.MaxBid = 0
	c.TaoPrice = nil
	c.MinerNodesErrors = map[string]string{}
}
