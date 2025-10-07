package callbacks

import (
	"fmt"
	"net"
	"sync"

	"targon/internal/cvm"
	"targon/internal/targon"

	"github.com/manifold-inc/manifold-sdk/lib/utils"
)

// Gets all nodes and adds them to the core
func getNodesAll(c *targon.Core) {
	wg := sync.WaitGroup{}
	wg.Add(len(c.Neurons))
	mu := sync.Mutex{}
	c.Deps.Log.Infof("Checking CVM nodes for %d miners", len(c.Neurons))
	attester := cvm.NewAttester(c.Deps.Env.TimeoutMult, c.Deps.Hotkey, c.Deps.Env.TowerURL)
	totalNodes := 0
	for _, n := range c.Neurons {
		uid := fmt.Sprintf("%d", n.UID.Int64())
		mu.Lock()
		if c.MinerErrors[uid] == nil {
			c.MinerErrors[uid] = make(map[string]string)
		}
		mu.Unlock()
		go func() {
			defer wg.Done()

			// Inactive miner check
			var neuronIPAddr net.IP = n.AxonInfo.IP.Bytes()
			if n.AxonInfo.IP.String() == "0" {
				return
			}
			nodes, err := attester.GetNodes(utils.AccountIDToSS58(n.Hotkey), fmt.Sprintf("%s:%d", neuronIPAddr.String(), n.AxonInfo.Port))

			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				// supress this in prod; we can always check mongo for errors
				c.Deps.Log.Debugw("error getting miner nodes", "uid", uid, "error", err)
				c.MinerErrors[uid][n.AxonInfo.IP.String()] = err.Error()
				return
			}

			c.MinerNodes[uid] = nodes
			delete(c.MinerErrors[uid], n.AxonInfo.IP.String())
			totalNodes += len(nodes)
		}()
	}
	wg.Wait()
	c.Deps.Log.Infof("Found %d miners with a total of %d nodes", len(c.MinerNodes), totalNodes)
}
