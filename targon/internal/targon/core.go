// Package targon
package targon

import (
	"targon/internal/setup"
	"targon/internal/tower"

	"github.com/subtrahend-labs/gobt/runtime"
)

type MinerNode struct {
	IP    string `json:"ip"`
	Price int    `json:"price"`
}

type AccountInfo struct {
	UID   string `bson:"uid"`
	Alpha int64  `bson:"alpha"`
}

type Core struct {
	Neurons          map[string]runtime.NeuronInfo `bson:"-"`
	Deps             *setup.Dependencies           `bson:"-"`
	HotkeyToUID      map[string]string             `bson:"hotkey_to_uid"`
	ColdkeyToUID     map[string]AccountInfo        `bson:"coldkey_to_account_info"`
	BurnDistribution map[int]int                   `bson:"burn_distribution"`

	// uid -> nodes
	MinerNodes map[string][]*MinerNode `bson:"miner_nodes"`

	// uid -> ip/location -> error
	MinerErrors map[string]map[string]string `bson:"miner_errors"`

	// uid -> nodes -> gpus
	VerifiedNodes map[string]map[string]*UserData `bson:"verified_nodes"`

	// node id -> seen
	NodeIds map[string]bool `bson:"node_ids"`

	// Total tao emission pool for mieners (in usd)
	EmissionPool   *float64                 `bson:"emission_pool"`
	Auctions       map[string]tower.Auction `bson:"auctions"`
	AuctionResults map[string][]*MinerBid   `bson:"auction_results"`
	TaoPrice       *float64                 `bson:"tao_price"`
	StartupBlock   int                      `bson:"startup_block"`
}

func CreateCore(d *setup.Dependencies) *Core {
	// TODO init maps
	return &Core{
		Deps:          d,
		MinerNodes:    map[string][]*MinerNode{},
		VerifiedNodes: map[string]map[string]*UserData{},
		Neurons:       map[string]runtime.NeuronInfo{},
		NodeIds:       map[string]bool{},
		MinerErrors:   map[string]map[string]string{},
	}
}
