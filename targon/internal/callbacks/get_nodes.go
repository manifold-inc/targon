package callbacks

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"sync"
	"time"

	"targon/internal/subtensor/utils"
	"targon/internal/targon"
	errutil "targon/internal/utils"

	"github.com/centrifuge/go-substrate-rpc-client/v4/signature"
	"github.com/subtrahend-labs/gobt/boilerplate"
	"github.com/subtrahend-labs/gobt/runtime"
)

func getNodesAll(c *targon.Core) {
	wg := sync.WaitGroup{}
	wg.Add(len(c.Neurons))
	c.Deps.Log.Infof("Checking CVM nodes for %d miners", len(c.Neurons))
	totalNodes := 0
	for _, n := range c.Neurons {
		uid := fmt.Sprintf("%d", n.UID.Int64())
		go func() {
			defer wg.Done()
			nodes, err := getNodes(c.Deps.Env.TIMEOUT_MULT, c.Deps.Hotkey, &n)
			c.Mnmu.Lock()
			defer c.Mnmu.Unlock()

			if err != nil {
				// supress this in prod; we can always check mongo for errors
				c.Deps.Log.Debugw("error getting miner nodes", "uid", uid, "error", err)
				c.MinerNodesErrors[uid] = err.Error()
				return
			}

			// Max price is max bid, min price is 1
			for _, v := range nodes {
				if v.Price == 0 {
					v.Price = c.MaxBid
				}
				v.Price = max(min(v.Price, c.MaxBid), 1)
			}
			c.MinerNodes[uid] = nodes
			delete(c.MinerNodesErrors, uid)
			totalNodes += len(nodes)
		}()
	}
	wg.Wait()
	c.Deps.Log.Infof("Found %d miners with a total of %d nodes", len(c.MinerNodes), totalNodes)
}

func getNodes(timeout_mult time.Duration, hotkey signature.KeyringPair, n *runtime.NeuronInfo) ([]*targon.MinerNode, error) {
	if n.AxonInfo.IP.String() == "0" {
		err := errors.New("inactive miner")
		return nil, err
	}
	tr := &http.Transport{
		TLSHandshakeTimeout: 5 * time.Second * timeout_mult,
		MaxConnsPerHost:     1,
		DisableKeepAlives:   true,
	}
	client := &http.Client{Transport: tr, Timeout: 5 * time.Second * timeout_mult}
	var neuronIpAddr net.IP = n.AxonInfo.IP.Bytes()
	req, err := http.NewRequest(
		"GET",
		fmt.Sprintf("http://%s:%d/cvm", neuronIpAddr, n.AxonInfo.Port),
		nil,
	)
	if err != nil {
		return nil, errutil.Wrap("failed to generate request to miner", err)
	}
	headers, err := boilerplate.GetEpistulaHeaders(
		hotkey,
		utils.AccountIDToSS58(n.Hotkey),
		[]byte{},
	)
	if err != nil {
		return nil, errutil.Wrap("failed generating epistula headers", err)
	}
	for key, value := range headers {
		req.Header.Set(key, value)
	}
	req.Close = true
	resp, err := client.Do(req)
	if err != nil {
		return nil, errutil.Wrap("failed sending request to miner", err)
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("bad status code %d", resp.StatusCode)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, errutil.Wrap("failed reading miner response", err)
	}

	// Backwards compat; remove later on
	var nodesv2 []*targon.MinerNode
	var nodesv1 []string
	err = json.Unmarshal(body, &nodesv2)
	if err != nil {
		// reset this encase it got accidentally populated by the previous unmarshal
		nodesv2 = []*targon.MinerNode{}
		err = json.Unmarshal(body, &nodesv1)
		if err != nil {
			return nil, errutil.Wrap("failed reading miner response", err)
		}
		for _, node := range nodesv1 {
			nodesv2 = append(nodesv2, &targon.MinerNode{
				Ip:    node,
				Price: 0,
			})
		}
	}
	return nodesv2, nil
}
