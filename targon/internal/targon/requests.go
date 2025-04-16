package targon

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"

	"targon/internal/subtensor/utils"
	"targon/validator"

	"github.com/google/uuid"
	"github.com/subtrahend-labs/gobt/runtime"
)

func GetCVMNodes(c *Core, client *http.Client, n *runtime.NeuronInfo) ([]string, error) {
	uid := fmt.Sprintf("%d", n.UID.Int64())
	req, err := http.NewRequest("GET", fmt.Sprintf("http://%s:%d/cvm", n.AxonInfo.IP, n.AxonInfo.Port), nil)
	if err != nil {
		c.Deps.Log.Warnw("Failed to generate request to miner", "error", err, "uid", uid)
		return nil, err
	}
	headers, err := validator.GetEpistulaHeaders(c.Deps.Env.HOTKEY_SS58, c.Deps.Env.HOTKEY_PUBLIC_KEY, c.Deps.Env.HOTKEY_PRIVATE_KEY, utils.AccountIDToSS58(n.Hotkey), []byte{})
	if err != nil {
		c.Deps.Log.Warnw("Failed generating epistula headers", "error", err, "uid", uid)
		return nil, err
	}
	for key, value := range headers {
		req.Header.Set(key, value)
	}
	req.Close = true
	resp, err := client.Do(req)
	if err != nil {
		c.Deps.Log.Debugw("Failed sending request to miner", "error", err, "uid", uid)
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		c.Deps.Log.Debugw("Miner sent back unexpected status", "status", fmt.Sprintf("%d", resp.StatusCode), "uid", uid)
		return nil, fmt.Errorf("bad status code %d", resp.StatusCode)
	}
	var nodes []string
	err = json.NewDecoder(resp.Body).Decode(&nodes)
	if err != nil {
		c.Deps.Log.Debugw("Failed reading miner response", "error", err, "uid", uid)
		return nil, err
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

			// TODO this should return gpu information we can use for scoring
			ok = CheckCVMAttest(c, client, n, node)
			if !ok {
				return
			}
			nmu.Lock()
			passingNodes = append(passingNodes, node)
			nmu.Unlock()
		}()
	}
	nwg.Wait()

	c.NeuronNodesMu.Lock()
	defer c.NeuronNodesMu.Unlock()
	c.NeuronNodes[fmt.Sprintf("%d", n.UID.Int64())] = passingNodes
	return passingNodes, nil
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
		c.Deps.Log.Warnw("Failed sending request to miner", "error", err, "uid", uid)
		return false
	}
	if resp.StatusCode != http.StatusOK {
		return false
	}
	return true
}

type AttestBody struct {
	Nonce string `json:"nonce"`
}

func CheckCVMAttest(c *Core, client *http.Client, n *runtime.NeuronInfo, cvmIP string) bool {
	uid := fmt.Sprintf("%d", n.UID.Int64())
	h1 := strings.ReplaceAll(uuid.NewString(), "-", "")
	h2 := strings.ReplaceAll(uuid.NewString(), "-", "")
	nonce := h1 + h2
	data := AttestBody{Nonce: nonce}
	body, _ := json.Marshal(data)
	req, err := http.NewRequest("POST", fmt.Sprintf("http://%s/api/v1/attest", cvmIP), bytes.NewBuffer(body))
	if err != nil {
		c.Deps.Log.Warnw("Failed to generate request to miner", "error", err, "uid", uid)
		return false
	}
	headers, err := validator.GetEpistulaHeaders(c.Deps.Env.HOTKEY_SS58, c.Deps.Env.HOTKEY_PUBLIC_KEY, c.Deps.Env.HOTKEY_PRIVATE_KEY, utils.AccountIDToSS58(n.Hotkey), body)
	if err != nil {
		c.Deps.Log.Warnw("Failed generating epistula headers", "error", err, "uid", uid)
		return false
	}

	req.Header.Set("Content-Type", "application/json")
	for key, value := range headers {
		req.Header.Set(key, value)
	}
	req.Close = true
	resp, err := client.Do(req)
	if err != nil {
		c.Deps.Log.Warnw("Failed sending request to miner", "error", err, "uid", uid)
		return false
	}
	if resp.StatusCode != http.StatusOK {
		c.Deps.Log.Debugw("Bad status code from miner attest", "status", fmt.Sprintf("%d", resp.StatusCode))
		return false
	}
	resBody, err := io.ReadAll(resp.Body)
	if err != nil {
		c.Deps.Log.Debugw("Failed reading response", "error", err)
		return false
	}
	var attestRes map[string]any
	err = json.Unmarshal(resBody, &attestRes)
	if err != nil {
		c.Deps.Log.Debugw("Failed unmarshaling response", "error", err)
		return false
	}
	// TODO send to sidecar for checking w/ nvidia
	return true
}
