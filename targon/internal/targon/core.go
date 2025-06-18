package targon

import (
	"sync"

	"targon/internal/setup"

	"github.com/subtrahend-labs/gobt/runtime"
)

type Core struct {
	Neurons map[string]runtime.NeuronInfo `bson:"-"`
	Deps    *setup.Dependencies           `bson:"-"`
	Mnmu    sync.Mutex                    `bson:"-"`
	// uid -> nodes
	MinerNodes map[string][]string `bson:"miner_nodes,omitempty"`
	Hpmu              sync.Mutex                   `bson:"-"`
	// uid -> nodes -> []passed
	HealthcheckPasses map[string]map[string][]bool `bson:"healthcheck_passes,omitempty"`
	// uid -> nodes -> gpus
	PassedAttestation map[string]map[string][]string `bson:"passed_attestation,omitempty"`
	// uid -> nodes -> icons
	ICONS map[string]map[string]string `bson:"icons,omitempty"`
	// gpu id -> seen
	GPUids map[string]bool `bson:"gp_uids,omitempty"`
	// SN Emission
	EmissionPool *float64 `bson:"emission_pool,omitempty"`
	TaoPrice     *float64 `bson:"tao_price,omitempty"`

	// Global core lock
	Mu sync.Mutex `bson:"-"`
}

func CreateCore(d *setup.Dependencies) *Core {
	// TODO init maps
	return &Core{
		Deps:              d,
		MinerNodes:        map[string][]string{},
		HealthcheckPasses: map[string]map[string][]bool{},
		PassedAttestation: map[string]map[string][]string{},
		ICONS:             map[string]map[string]string{},
		Neurons:           map[string]runtime.NeuronInfo{},
		GPUids:            map[string]bool{},
	}
}
