package callbacks

import (
	"fmt"

	"targon/internal/targon"

	"github.com/centrifuge/go-substrate-rpc-client/v4/types"
	"github.com/subtrahend-labs/gobt/boilerplate"
	"github.com/subtrahend-labs/gobt/runtime"
)

func getNeuronsCallback(v *boilerplate.BaseChainSubscriber, c *targon.Core, h types.Header) {
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
	for _, n := range neurons {
		uid := fmt.Sprintf("%d", n.UID.Int64())
		c.Neurons[uid] = n
	}
	c.Deps.Log.Info("Neurons Updated")
}
