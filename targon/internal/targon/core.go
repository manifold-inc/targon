package targon

import (
	"sync"

	"targon/internal/setup"

	"github.com/subtrahend-labs/gobt/runtime"
)

type MinerNode struct {
	Ip    string `json:"ip"`
	Price int    `json:"price"`
}

type Core struct {
	Neurons map[string]runtime.NeuronInfo `bson:"-"`
	Deps    *setup.Dependencies           `bson:"-"`
	Mnmu    sync.Mutex                    `bson:"-"`

	HotkeyToUid map[string]string `bson:"hotkey_to_uid"`

	// uid -> nodes
	MinerNodes       map[string][]*MinerNode `bson:"miner_nodes"`
	MinerNodesErrors map[string]string       `bson:"miner_nodes_errors"`
	Hpmu             sync.Mutex              `bson:"-"`
	// uid -> nodes -> gpus
	PassedAttestation map[string]map[string][]string `bson:"passed_attestation"`
	AttestErrors      map[string]map[string]string   `bson:"attest_errors"`
	// gpu id -> seen
	GPUids map[string]bool `bson:"gp_uids"`
	// Total tao emission pool for mieners
	EmissionPool   *float64               `bson:"emission_pool"`
	Auctions       map[string]int         `bson:"auctions"`
	AuctionResults map[string][]*MinerBid `bson:"auction_results"`
	MaxBid         int                    `bson:"max_bid"`
	TaoPrice       *float64               `bson:"tao_price"`
	StartupBlock   int                    `bson:"startup_block"`

	// Global core locks
	Mu sync.Mutex `bson:"-"`
}

func CreateCore(d *setup.Dependencies) *Core {
	// TODO init maps
	return &Core{
		Deps:              d,
		MinerNodes:        map[string][]*MinerNode{},
		PassedAttestation: map[string]map[string][]string{},
		Neurons:           map[string]runtime.NeuronInfo{},
		GPUids:            map[string]bool{},
		MinerNodesErrors:  map[string]string{},
		AttestErrors:      map[string]map[string]string{},
	}
}
