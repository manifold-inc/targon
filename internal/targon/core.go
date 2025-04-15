package targon

import (
	"sync"

	"targon/internal/setup"

	"github.com/subtrahend-labs/gobt/runtime"
)

type Core struct {
	Neurons       []runtime.NeuronInfo
	NeuronsMu     sync.RWMutex
	Deps          *setup.Dependencies
	NeuronNodes   map[string][]string
	NeuronNodesMu sync.Mutex
}

func CreateCore(d *setup.Dependencies) *Core {
	return &Core{Deps: d}
}
