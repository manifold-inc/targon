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
		TLSHandshakeTimeout: 5 * time.Second,
		MaxConnsPerHost:     1,
		DisableKeepAlives:   true,
	}
	client := &http.Client{Transport: tr, Timeout: 5 * time.Second}
	wg := sync.WaitGroup{}
	wg.Add(len(c.Neurons))
	c.Deps.Log.Infof("Checking CVM nodes for %d miners", len(c.Neurons))
	totalNodes := 0
	for _, n := range c.Neurons {
		go func() {
			uid := fmt.Sprintf("%d", n.UID.Int64())
			defer wg.Done()
			nodes, err := cvm.GetNodes(c, client, &n)
			// Either nil or list or strings,
			// errors are logged internally, dont need to handle them here
			c.Mnmu.Lock()
			defer c.Mnmu.Unlock()
			c.MinerNodes[uid] = nodes
			if err != nil {
				return
			}
			totalNodes += len(nodes)
		}()
	}
	wg.Wait()
	c.Deps.Log.Infof("Found %d miners with a total of %d nodes", len(c.MinerNodes), totalNodes)
}
