package targon

import (
	"context"
	"errors"
	"time"

	"targon/internal/setup"

	"github.com/centrifuge/go-substrate-rpc-client/v4/types"
	"go.mongodb.org/mongo-driver/v2/bson"
	"go.mongodb.org/mongo-driver/v2/mongo/options"
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
		Core:      c,
		Block:     int(h.Number),
		Timestamp: time.Now().Unix(),
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

type Backup struct {
	Core      *Core `bson:"inline"`
	Timestamp int64 `bson:"timestamp,omitempty"`
}

func SaveMongoBackup(c *Core) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	backup := Backup{Core: c, Timestamp: time.Now().Unix()}

	// Store miner info with emissions data
	minerInfoCol := c.Deps.Mongo.Database("targon").Collection("miner_info_backup")
	_, err := minerInfoCol.InsertOne(ctx, backup)
	return err
}

func LoadMongoBackup(c *Core) error {
	minerCol := c.Deps.Mongo.Database("targon").Collection("miner_info_backup")
	opts := options.FindOne().SetSort(bson.D{{Key: "timestamp", Value: -1}})

	// Find the record with the max value
	var r Backup
	err := minerCol.FindOne(context.TODO(), bson.D{}, opts).Decode(&r)
	if err != nil {
		return err
	}
	then := time.Unix(r.Timestamp, 0)
	if time.Since(then) > 30*time.Minute {
		return errors.New("Backup is too stale")
	}
	if r.Core.MinerNodes != nil {
		c.MinerNodes = r.Core.MinerNodes
	}
	if r.Core.MinerNodesErrors != nil {
		c.MinerNodesErrors = r.Core.MinerNodesErrors
	}
	if r.Core.HealthcheckPasses != nil {
		c.HealthcheckPasses = r.Core.HealthcheckPasses
	}
	if r.Core.PassedAttestation != nil {
		c.PassedAttestation = r.Core.PassedAttestation
	}
	if r.Core.AttestErrors != nil {
		c.AttestErrors = r.Core.AttestErrors
	}
	if r.Core.GPUids != nil {
		c.GPUids = r.Core.GPUids
	}
	if r.Core.EmissionPool != nil {
		c.EmissionPool = r.Core.EmissionPool
	}
	if r.Core.Auctions != nil {
		c.Auctions = r.Core.Auctions
	}
	c.MaxBid = r.Core.MaxBid
	c.TaoPrice = r.Core.TaoPrice
	return nil
}
