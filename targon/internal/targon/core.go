package targon

import (
	"sync"

	"targon/internal/setup"

	"github.com/subtrahend-labs/gobt/runtime"
)

type Core struct {
	Neurons map[string]runtime.NeuronInfo
	Deps    *setup.Dependencies
	// uid -> nodes
	Mnmu       sync.Mutex
	MinerNodes map[string][]string
	// uid -> nodes -> []passed
	Hpmu              sync.Mutex
	HealthcheckPasses map[string]map[string][]bool
	// uid -> nodes -> gpus
	PassedAttestation map[string]map[string][]string
	// gpu id -> seen
	GPUids map[string]bool
	// SN Emission
	EmissionPool *float64
	TaoPrice *float64

	// Global core lock
	mu sync.Mutex
}

func CreateCore(d *setup.Dependencies) *Core {
	// TODO init maps
	return &Core{
		Deps:              d,
		MinerNodes:        map[string][]string{},
		HealthcheckPasses: map[string]map[string][]bool{},
		PassedAttestation: map[string]map[string][]string{},
		Neurons:           map[string]runtime.NeuronInfo{},
		GPUids:            map[string]bool{},
	}
}
