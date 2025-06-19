package targon

import (
	"context"
	"errors"
	"time"

	"targon/internal/setup"

	"github.com/centrifuge/go-substrate-rpc-client/v4/types"
)

func SyncMongo(c *Core, uids, scores []types.U16, h types.Header) error {
	incentives := make([]float64, len(scores))
	for i, score := range scores {
		incentives[i] = (float64(score) / float64(setup.U16MAX))
	}

	uidsInt := make([]uint16, len(uids))
	for i, uid := range uids {
		uidsInt[i] = uint16(uid)
	}

	minerInfo := MinerInfo{
		Core:         c,
		Block:        int(h.Number),
		EmissionPool: *c.EmissionPool,
		TaoPrice:     *c.TaoPrice,
		Timestamp:    time.Now().Unix(),
		Weights: Weights{
			UIDs:       uidsInt,
			Incentives: incentives,
		},
	}

	if c.Deps.Mongo == nil {
		return errors.New("no mongo client")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Store miner info with emissions data
	minerInfoCol := c.Deps.Mongo.Database("targon").Collection("miner_info")
	_, err := minerInfoCol.InsertOne(ctx, minerInfo)
	return err
}
