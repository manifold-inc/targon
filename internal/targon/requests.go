package targon

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sync"

	"targon/internal/subtensor/utils"
	"targon/validator"

	"github.com/subtrahend-labs/gobt/runtime"
)

func GetCVMNodes(c *Core, client *http.Client, n *runtime.NeuronInfo) {
	uid := fmt.Sprintf("%d", n.UID.Int64())
	req, err := http.NewRequest("GET", fmt.Sprintf("http://%s:%d/cvm", n.AxonInfo.IP, n.AxonInfo.Port), nil)
	if err != nil {
		c.Deps.Log.Warnw("Failed to generate request to miner", "error", err, "uid", uid)
		return
	}
	headers, err := validator.GetEpistulaHeaders(c.Deps.Env.HOTKEY_SS58, c.Deps.Env.HOTKEY_PUBLIC_KEY, c.Deps.Env.HOTKEY_PRIVATE_KEY, utils.AccountIDToSS58(n.Hotkey), []byte{})
	if err != nil {
		c.Deps.Log.Warnw("Failed generating epistula headers", "error", err, "uid", uid)
		return
	}
	for key, value := range headers {
		req.Header.Set(key, value)
	}
	req.Close = true
	resp, err := client.Do(req)
	if err != nil {
		c.Deps.Log.Warnw("Failed reading miner response", "error", err, "uid", uid)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		c.Deps.Log.Warnw("Miner sent back unexpected status", "status", fmt.Sprintf("%d", resp.StatusCode), "uid", uid)
		return
	}
	var nodes []string
	err = json.NewDecoder(resp.Body).Decode(&nodes)
	if err != nil {
		c.Deps.Log.Warnw("Failed reading miner response", "error", err, "uid", uid)
		return
	}
	nwg := sync.WaitGroup{}
	nwg.Add(len(nodes))
	passingNodes := []string{}
	nmu := sync.Mutex{}
	for _, node := range nodes {
		go func() {
			defer nwg.Done()
			ok := CheckCVMHealth(c, client, n, node)
			if !ok {
				return
			}
			nmu.Lock()
			passingNodes = append(passingNodes, node)
			nmu.Unlock()
		}()
	}
	nwg.Wait()

	// Dont bother setting arrays for miners that dont pass any
	if len(passingNodes) == 0 {
		return
	}
	c.NeuronNodesMu.Lock()
	defer c.NeuronNodesMu.Unlock()
	c.NeuronNodes[fmt.Sprintf("%d", n.UID.Int64())] = passingNodes
}

func CheckCVMHealth(c *Core, client *http.Client, n *runtime.NeuronInfo, cvmIP string) bool {
	uid := fmt.Sprintf("%d", n.UID.Int64())
	req, err := http.NewRequest("GET", fmt.Sprintf("http://%s/health", cvmIP), nil)
	if err != nil {
		c.Deps.Log.Warnw("Failed to generate request to miner", "error", err, "uid", uid)
		return false
	}
	headers, err := validator.GetEpistulaHeaders(c.Deps.Env.HOTKEY_SS58, c.Deps.Env.HOTKEY_PUBLIC_KEY, c.Deps.Env.HOTKEY_PRIVATE_KEY, utils.AccountIDToSS58(n.Hotkey), []byte{})
	if err != nil {
		c.Deps.Log.Warnw("Failed generating epistula headers", "error", err, "uid", uid)
		return false
	}
	for key, value := range headers {
		req.Header.Set(key, value)
	}
	req.Close = true
	resp, err := client.Do(req)
	if err != nil {
		c.Deps.Log.Warnw("Failed to generate request to miner", "error", err, "uid", uid)
		return false
	}
	if resp.StatusCode != http.StatusOK {
		return false
	}
	return true
}
