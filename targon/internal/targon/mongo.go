package targon

import (
	"context"
	"errors"
	"time"
)

func SyncMongo(core *Core, block int) error {
	if core.Deps.Mongo == nil {
		return errors.New("no mongo client")
	}
	minerInfo := MinerInfo{Core: core, Block: block}
	minerInfoCol := core.Deps.Mongo.Database("targon").Collection("miner_info")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	_, _ = minerInfoCol.InsertOne(ctx, minerInfo)
	return nil
}
