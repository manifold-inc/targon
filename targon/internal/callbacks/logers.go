package callbacks

import (
	"fmt"

	"targon/internal/targon"

	"github.com/centrifuge/go-substrate-rpc-client/v4/types"
)

func logWeights(c *targon.Core) {
	uids, scores, _, _ := getWeights(c)
	c.Deps.Log.Infow(
		"Current Weights",
		"uids",
		fmt.Sprintf("%+v", uids),
		"scores",
		fmt.Sprintf("%+v", scores),
	)
}

func logBlockCallback(c *targon.Core, h types.Header) {
	// Run Every Block
	c.Deps.Log.Infow(
		"New block",
		"block",
		fmt.Sprintf("%v", h.Number),
		"left_in_interval",
		fmt.Sprintf("%d", 360-(h.Number%360)),
	)
}
