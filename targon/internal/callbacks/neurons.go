package callbacks

import (
	"fmt"

	"targon/internal/targon"

	"github.com/centrifuge/go-substrate-rpc-client/v4/types"
	"github.com/manifold-inc/manifold-sdk/lib/utils"
	"github.com/subtrahend-labs/gobt/runtime"
)

func getNeuronsCallback(c *targon.Core, h types.Header) {
	c.Deps.Log.Info("Updating neurons")
	neurons, err := runtime.GetNeurons(c.Deps.Client, uint16(c.Deps.Env.Netuid), &h.ParentHash)
	if err != nil {
		blockHash, err := c.Deps.Client.Api.RPC.Chain.GetBlockHashLatest()
		if err != nil {
			c.Deps.Log.Errorw("Failed getting blockhash for neurons", "error", err)
			return
		}
		neurons, err = runtime.GetNeurons(c.Deps.Client, uint16(c.Deps.Env.Netuid), &blockHash)
		if err != nil {
			c.Deps.Log.Errorw("Failed getting neurons", "error", err)
			return
		}
	}
	c.HotkeyToUID = make(map[string]string)
	c.ColdkeyToUID = make(map[string]string)
	for _, n := range neurons {
		uid := fmt.Sprintf("%d", n.UID.Int64())
		c.Neurons[uid] = n
		c.HotkeyToUID[utils.AccountIDToSS58(n.Hotkey)] = uid
		c.ColdkeyToUID[utils.AccountIDToSS58(n.Coldkey)] = uid
	}
	c.Deps.Log.Info("Neurons Updated")
}
