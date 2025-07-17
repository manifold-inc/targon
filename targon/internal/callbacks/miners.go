package callbacks

import (
	"fmt"
	"net/http"
	"sync"
	"time"

	"targon/internal/cvm"
	"targon/internal/targon"
)

func getMinerNodes(c *targon.Core) {
	tr := &http.Transport{
		TLSHandshakeTimeout: 5 * time.Second * c.Deps.Env.TIMEOUT_MULT,
		MaxConnsPerHost:     1,
		DisableKeepAlives:   true,
	}
	client := &http.Client{Transport: tr, Timeout: 5 * time.Second * c.Deps.Env.TIMEOUT_MULT}
	wg := sync.WaitGroup{}
	wg.Add(len(c.Neurons))
	c.Deps.Log.Infof("Checking CVM nodes for %d miners", len(c.Neurons))
	totalNodes := 0
	for _, n := range c.Neurons {
		uid := fmt.Sprintf("%d", n.UID.Int64())

		go func() {
			defer wg.Done()
			nodes, err := cvm.GetNodes(c, client, &n)
			c.Mnmu.Lock()
			defer c.Mnmu.Unlock()
			if err == nil {
				c.MinerNodes[uid] = nodes
				totalNodes += len(nodes)
				delete(c.MinerNodesErrors, uid)
				return
			}
			// suppress this in prod; we can always check mongo for errors
			c.Deps.Log.Debugw("error getting miner nodes", "uid", uid, "error", err)
			c.MinerNodesErrors[uid] = err.Error()
		}()
	}
	wg.Wait()
	c.Deps.Log.Infof("Found %d miners with a total of %d nodes", len(c.MinerNodes), totalNodes)
}
