package targon

import (
	"context"
	"errors"
	"time"
)

func SyncMongo(core *Core, minerInfo MinerInfo) error {
	if core.Deps.Mongo == nil {
		return errors.New("no mongo client")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Store miner info with emissions data
	minerInfoCol := core.Deps.Mongo.Database("targon").Collection("miner_info")
	_, err := minerInfoCol.InsertOne(ctx, minerInfo)
	return err
}
