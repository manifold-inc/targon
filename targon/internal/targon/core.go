package targon

import (
	"sync"

	"targon/internal/setup"

	"github.com/subtrahend-labs/gobt/runtime"
)

type Core struct {
	Neurons          []runtime.NeuronInfo
	Deps             *setup.Dependencies
	NeuronHardware   map[string][]string
	// Global core lock
	mu               sync.Mutex
}

func CreateCore(d *setup.Dependencies) *Core {
	return &Core{Deps: d}
}
