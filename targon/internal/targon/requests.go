package targon

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"targon/internal/subtensor/utils"

	"github.com/google/uuid"
	"github.com/subtrahend-labs/gobt/boilerplate"
	"github.com/subtrahend-labs/gobt/runtime"
)

func GetCVMNodes(c *Core, client *http.Client, n *runtime.NeuronInfo) ([]string, error) {
	uid := fmt.Sprintf("%d", n.UID.Int64())
	Log := c.Deps.Log.With("uid", uid)
	if n.AxonInfo.IP.String() == "0" {
		err := errors.New("inactive miner")
		Log.Debug(err.Error())
		return nil, err

	}
	var neuronIpAddr net.IP = n.AxonInfo.IP.Bytes()
	req, err := http.NewRequest("GET", fmt.Sprintf("http://%s:%d/cvm", neuronIpAddr, n.AxonInfo.Port), nil)
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

	if resp.StatusCode != http.StatusOK {
		Log.Debugw("Miner sent back unexpected status", "status", fmt.Sprintf("%d", resp.StatusCode))
		return nil, fmt.Errorf("bad status code %d", resp.StatusCode)
	}
	var nodes []string
	err = json.NewDecoder(resp.Body).Decode(&nodes)
	resp.Body.Close()
	if err != nil {
		Log.Debugw("Failed reading miner response", "error", err)
		return nil, err
	}
	nwg := sync.WaitGroup{}
	nwg.Add(len(nodes))
	gpusModels := []string{}
	nmu := sync.Mutex{}

	tr := &http.Transport{
		TLSHandshakeTimeout: 5 * time.Minute,
		MaxConnsPerHost:     1,
		DisableKeepAlives:   true,
	}
	attestClient := &http.Client{Transport: tr, Timeout: 5 * time.Minute}

	for _, node := range nodes {
		go func() {
			defer nwg.Done()
			ok := CheckCVMHealth(c, client, n, node)
			if !ok {
				c.Deps.Log.Infow("Failed healthcheck", "uid", uid)
				return
			}

			c.Deps.Log.Infow("Passed Healthcheck", "uid", uid)
			gpus, err := CheckCVMAttest(c, attestClient, n, node)
			if err != nil {
				c.Deps.Log.Infow("Failed Attest", "uid", uid, "error", err)
				return
			}
			c.Deps.Log.Infow("Passed CVM Attest", "uid", uid)
			nmu.Lock()
			gpusModels = append(gpusModels, gpus...)
			nmu.Unlock()
		}()
	}
	nwg.Wait()

	c.Deps.Log.Infow("Found gpu models for miner", "uid", uid)
	c.NeuronHardware[uid] = gpusModels

	return gpusModels, nil
}

func CheckCVMHealth(c *Core, client *http.Client, n *runtime.NeuronInfo, cvmIP string) bool {
	uid := fmt.Sprintf("%d", n.UID.Int64())
	Log := c.Deps.Log.With("uid", uid)
	cvmIP = strings.TrimPrefix(cvmIP, "http://")
	req, err := http.NewRequest("GET", fmt.Sprintf("http://%s/health", cvmIP), nil)
	if err != nil {
		Log.Debugw("Failed to generate request to miner", "error", err)
		return false
	}
	headers, err := boilerplate.GetEpistulaHeaders(c.Deps.Hotkey, utils.AccountIDToSS58(n.Hotkey), []byte{})
	if err != nil {
		Log.Debugw("Failed generating epistula headers", "error", err)
		return false
	}
	for key, value := range headers {
		req.Header.Set(key, value)
	}
	req.Close = true
	resp, err := client.Do(req)
	if err != nil {
		Log.Debugw("Failed sending request to miner", "error", err)
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
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
		Log.Debugw("Failed to generate request to miner", "error", err)
		return nil, err
	}
	headers, err := boilerplate.GetEpistulaHeaders(c.Deps.Hotkey, utils.AccountIDToSS58(n.Hotkey), body)
	if err != nil {
		Log.Debugw("Failed generating epistula headers", "error", err)
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	for key, value := range headers {
		req.Header.Set(key, value)
	}
	req.Close = true
	resp, err := client.Do(req)
	if err != nil {
		Log.Debugw("Failed sending request to miner", "error", err)
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		Log.Debugw("Bad status code from miner attest", "status", fmt.Sprintf("%d", resp.StatusCode))
		return nil, errors.New("Bad status code from miner attest: " + resp.Status)
	}
	resBody, err := io.ReadAll(resp.Body)
	resp.Body.Close()
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

	if !attestRes.GPU.AttestationResult {
		err = errors.New("gpu attestation failed")
		Log.Debug(err.Error())
		return nil, err
	}

	if !attestRes.GPU.Valid {
		err = errors.New("gpu attestation invalid")
		Log.Debug(err.Error())
		return nil, err
	}

	if !attestRes.Switch.AttestationResult {
		err = errors.New("switch attestation failed")
		Log.Debug(err.Error())
		return nil, err
	}

	if !attestRes.Switch.Valid {
		err = errors.New("switch attestation invalid")
		Log.Debug(err.Error())
		return nil, err
	}

	// Validate Attestation
	body, err = json.Marshal(map[string]any{
		"gpu":            attestRes.GPU,
		"switch":         attestRes.Switch,
		"expected_nonce": nonce,
	})
	if err != nil {
		Log.Debug("failed marshaling miner attest response", "error", err.Error())
		return nil, errors.New("failed marshaling miner attest response")
	}

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
		resp.Body.Close()
		Log.Warnw("Bad status code from nvidia-attest", "status", fmt.Sprintf("%d", resp.StatusCode))
		return nil, errors.New("Bad status code from miner attest: " + resp.Status)
	}
	resBody, err = io.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		Log.Warnw("Failed reading response body from nvidia-attest", "error", err)
		return nil, err
	}

	var attestResponse GPUAttestationResponse
	err = json.Unmarshal(resBody, &attestResponse)
	if err != nil {
		Log.Debugw("Failed decoding json response from nvidia-attest", "error", err)
		return nil, err
	}

	if !attestResponse.GPUAttestationSuccess || !attestResponse.SwitchAttestationSuccess {
		Log.Debugw("GPU or switch attestation failed",
			"gpu_success", attestResponse.GPUAttestationSuccess,
			"switch_success", attestResponse.SwitchAttestationSuccess)
		return nil, errors.New("GPU or switch attestation failed")
	}

	// Extract GPU types from the claims
	var gpuTypes []string
	if attestResponse.GPUClaims != nil {
		for _, claims := range attestResponse.GPUClaims {
			gpuTypes = append(gpuTypes, claims.GPUType)
			Log.Debugw("GPU attestation successful",
				"gpu_type", claims.GPUType)
		}
	}

	return gpuTypes, nil
}
