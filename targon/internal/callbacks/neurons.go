package callbacks

import (
	"fmt"

	"targon/internal/subtensor/utils"
	"targon/internal/targon"

	"github.com/centrifuge/go-substrate-rpc-client/v4/types"
	"github.com/subtrahend-labs/gobt/runtime"
)

func getNeuronsCallback(c *targon.Core, h types.Header) {
	c.Deps.Log.Info("Updating neurons")
	neurons, err := runtime.GetNeurons(c.Deps.Client, uint16(c.Deps.Env.NETUID), &h.ParentHash)
	if err != nil {
		blockHash, err := c.Deps.Client.Api.RPC.Chain.GetBlockHashLatest()
		if err != nil {
			c.Deps.Log.Errorw("Failed getting blockhash for neurons", "error", err)
			return
		}
		neurons, err = runtime.GetNeurons(c.Deps.Client, uint16(c.Deps.Env.NETUID), &blockHash)
		if err != nil {
			c.Deps.Log.Errorw("Failed getting neurons", "error", err)
			return
		}
	}
	c.HotkeyToUid = make(map[string]string)
	for _, n := range neurons {
		uid := fmt.Sprintf("%d", n.UID.Int64())
		c.Neurons[uid] = n
		c.HotkeyToUid[utils.AccountIDToSS58(n.Hotkey)] = uid
	}
	c.Deps.Log.Info("Neurons Updated")
}
