package callbacks

import (
	"targon/internal/targon"

	"github.com/subtrahend-labs/gobt/runtime"
)

func resetState(c *targon.Core) {
	c.Mu.Lock()
	defer c.Mu.Unlock()
	c.Neurons = make(map[string]runtime.NeuronInfo)
	c.MinerNodes = make(map[string][]string)
	c.GPUids = make(map[string]bool)
	// Dont really need to wipe tao price
	c.EmissionPool = nil
	// TODO maybe keep this alive longer than an interval
	c.HealthcheckPasses = make(map[string]map[string][]bool)
	c.PassedAttestation = make(map[string]map[string][]string)
	c.ICONS = make(map[string]map[string]string)
}
