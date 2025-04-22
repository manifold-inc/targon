package targon

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"

	"targon/internal/subtensor/utils"

	"github.com/google/uuid"
	"github.com/subtrahend-labs/gobt/boilerplate"
	"github.com/subtrahend-labs/gobt/runtime"
)

func GetCVMNodes(c *Core, client *http.Client, n *runtime.NeuronInfo) ([]string, error) {
	uid := fmt.Sprintf("%d", n.UID.Int64())
	Log := c.Deps.Log.With("uid", uid)
	req, err := http.NewRequest("GET", fmt.Sprintf("http://%s:%d/cvm", n.AxonInfo.IP, n.AxonInfo.Port), nil)
	if err != nil {
		Log.Warnw("Failed to generate request to miner", "error", err)
		return nil, err
	}
	headers, err := boilerplate.GetEpistulaHeaders(c.Deps.Hotkey, utils.AccountIDToSS58(n.Hotkey), []byte{})
	if err != nil {
		Log.Warnw("Failed generating epistula headers", "error", err)
		return nil, err
	}
	for key, value := range headers {
		req.Header.Set(key, value)
	}
	req.Close = true
	resp, err := client.Do(req)
	if err != nil {
		Log.Debugw("Failed sending request to miner", "error", err)
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		Log.Debugw("Miner sent back unexpected status", "status", fmt.Sprintf("%d", resp.StatusCode))
		return nil, fmt.Errorf("bad status code %d", resp.StatusCode)
	}
	var nodes []string
	err = json.NewDecoder(resp.Body).Decode(&nodes)
	if err != nil {
		Log.Debugw("Failed reading miner response", "error", err)
		return nil, err
	}
	nwg := sync.WaitGroup{}
	nwg.Add(len(nodes))
	gpusModels := []string{}
	nmu := sync.Mutex{}
	for _, node := range nodes {
		go func() {
			defer nwg.Done()
			ok := CheckCVMHealth(c, client, n, node)
			if !ok {
				return
			}

			gpus, err := CheckCVMAttest(c, client, n, node)
			if err != nil {
				return
			}
			nmu.Lock()
			gpusModels = append(gpusModels, gpus...)
			nmu.Unlock()
		}()
	}
	nwg.Wait()

	c.NeuronHardware[fmt.Sprintf("%d", n.UID.Int64())] = gpusModels
	return gpusModels, nil
}

func CheckCVMHealth(c *Core, client *http.Client, n *runtime.NeuronInfo, cvmIP string) bool {
	uid := fmt.Sprintf("%d", n.UID.Int64())
	Log := c.Deps.Log.With("uid", uid)
	req, err := http.NewRequest("GET", fmt.Sprintf("http://%s/health", cvmIP), nil)
	if err != nil {
		Log.Warnw("Failed to generate request to miner", "error", err)
		return false
	}
	headers, err := boilerplate.GetEpistulaHeaders(c.Deps.Hotkey, utils.AccountIDToSS58(n.Hotkey), []byte{})
	if err != nil {
		Log.Warnw("Failed generating epistula headers", "error", err)
		return false
	}
	for key, value := range headers {
		req.Header.Set(key, value)
	}
	req.Close = true
	resp, err := client.Do(req)
	if err != nil {
		Log.Warnw("Failed sending request to miner", "error", err)
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

func CheckCVMAttest(c *Core, client *http.Client, n *runtime.NeuronInfo, cvmIP string) ([]string, error) {
	uid := fmt.Sprintf("%d", n.UID.Int64())
	Log := c.Deps.Log.With("uid", uid)
	h1 := strings.ReplaceAll(uuid.NewString(), "-", "")
	h2 := strings.ReplaceAll(uuid.NewString(), "-", "")
	nonce := h1 + h2
	data := AttestBody{Nonce: nonce}
	body, _ := json.Marshal(data)
	req, err := http.NewRequest("POST", fmt.Sprintf("http://%s/api/v1/attest", cvmIP), bytes.NewBuffer(body))
	if err != nil {
		Log.Warnw("Failed to generate request to miner", "error", err)
		return nil, err
	}
	headers, err := boilerplate.GetEpistulaHeaders(c.Deps.Hotkey, utils.AccountIDToSS58(n.Hotkey), body)
	if err != nil {
		Log.Warnw("Failed generating epistula headers", "error", err)
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	for key, value := range headers {
		req.Header.Set(key, value)
	}
	req.Close = true
	resp, err := client.Do(req)
	if err != nil {
		Log.Warnw("Failed sending request to miner", "error", err, "uid")
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		Log.Debugw("Bad status code from miner attest", "status", fmt.Sprintf("%d", resp.StatusCode))
		return nil, errors.New("Bad status code from miner attest: " + resp.Status)
	}
	resBody, err := io.ReadAll(resp.Body)
	if err != nil {
		Log.Debugw("Failed reading response", "error", err)
		return nil, err
	}
	var attestRes AttestResponse
	err = json.Unmarshal(resBody, &attestRes)
	if err != nil {
		Log.Debugw("Failed unmarshaling response", "error", err)
		return nil, err
	}

	// Validate Attestation
	body, _ = json.Marshal(map[string]any{"data": attestRes, "expected_nonce": nonce})
	req, err = http.NewRequest("POST", fmt.Sprintf("%s/attest", c.Deps.Env.NVIDIA_ATTEST_ENDPOINT), bytes.NewBuffer(body))
	if err != nil {
		Log.Warnw("Failed to generate request to nvidia-attest", "error", err)
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Close = true
	resp, err = client.Do(req)
	if err != nil {
		Log.Warnw("Failed sending request to nvidia-attest", "error", err)
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		Log.Debugw("Bad status code from nvidia-attest", "status", fmt.Sprintf("%d", resp.StatusCode))
		return nil, errors.New("Bad status code from miner attest: " + resp.Status)
	}
	resBody, err = io.ReadAll(resp.Body)
	if err != nil {
		Log.Debugw("Failed reading response body from nvidia-attest", "error", err)
		return nil, err
	}
	isValid := false
	err = json.Unmarshal(resBody, &isValid)
	if err != nil {
		Log.Debugw("Failed decoding json response from nvidia-attest", "error", err)
		return nil, err
	}
	if !isValid {
		Log.Debugw("GPU information did not pass verification")
		return nil, errors.New("GPU Information did not pass verification")
	}
	gpus := []string{}
	for _, gi := range attestRes.GPUs {
		gpus = append(gpus, gi.Model)
	}
	if len(gpus) == 0 {
		err := errors.New("no gpus in attest report")
		Log.Debugw(err.Error())
		return nil, err
	}
	return gpus, nil
}
