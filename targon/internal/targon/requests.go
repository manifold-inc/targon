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

	"targon/internal/subtensor/utils"

	"github.com/google/uuid"
	"github.com/subtrahend-labs/gobt/boilerplate"
	"github.com/subtrahend-labs/gobt/runtime"
	"go.uber.org/zap"
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
	req, err := http.NewRequest(
		"GET",
		fmt.Sprintf("http://%s:%d/cvm", neuronIpAddr, n.AxonInfo.Port),
		nil,
	)
	if err != nil {
		Log.Warnw("Failed to generate request to miner", "error", err)
		return nil, err
	}
	headers, err := boilerplate.GetEpistulaHeaders(
		c.Deps.Hotkey,
		utils.AccountIDToSS58(n.Hotkey),
		[]byte{},
	)
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
		Log.Debugw(
			"Miner sent back unexpected status",
			"status",
			fmt.Sprintf("%d", resp.StatusCode),
		)
		return nil, fmt.Errorf("bad status code %d", resp.StatusCode)
	}
	var nodes []string
	err = json.NewDecoder(resp.Body).Decode(&nodes)
	if err != nil {
		Log.Debugw("Failed reading miner response", "error", err)
		return nil, err
	}
	return nodes, nil
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
	headers, err := boilerplate.GetEpistulaHeaders(
		c.Deps.Hotkey,
		utils.AccountIDToSS58(n.Hotkey),
		[]byte{},
	)
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

func getCVMAttestFromNode(
	c *Core,
	client *http.Client,
	n *runtime.NeuronInfo,
	cvmIP string,
	log *zap.SugaredLogger,
	nonce string,
) (*AttestResponse, error) {
	data := AttestBody{Nonce: nonce}
	body, _ := json.Marshal(data)
	req, err := http.NewRequest(
		"POST",
		fmt.Sprintf("http://%s/api/v1/attest", cvmIP),
		bytes.NewBuffer(body),
	)
	if err != nil {
		log.Debugw("Failed to generate request to miner", "error", err)
		return nil, err
	}
	headers, err := boilerplate.GetEpistulaHeaders(
		c.Deps.Hotkey,
		utils.AccountIDToSS58(n.Hotkey),
		body,
	)
	if err != nil {
		log.Debugw("Failed generating epistula headers", "error", err)
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	for key, value := range headers {
		req.Header.Set(key, value)
	}
	req.Close = true
	resp, err := client.Do(req)
	if err != nil {
		log.Debugw("Failed sending request to miner", "error", err)
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		log.Debugw(
			"Bad status code from miner attest",
			"status",
			fmt.Sprintf("%d", resp.StatusCode),
		)
		return nil, errors.New("Bad status code from miner attest: " + resp.Status)
	}
	resBody, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Debugw("Failed reading response", "error", err)
		return nil, err
	}
	var attestRes AttestResponse
	err = json.Unmarshal(resBody, &attestRes)
	if err != nil {
		log.Debugw("Failed unmarshaling response", "error", err)
		return nil, err
	}
	return &attestRes, nil
}

func verifyAttestResponse(
	c *Core,
	client *http.Client,
	attestRes *AttestResponse,
	nonce string,
	log *zap.SugaredLogger,
) (*GPUAttestationResponse, error) {
	// Validate Attestation
	body, err := json.Marshal(map[string]any{
		"gpu_remote":         attestRes.GPURemote,
		"switch_remote":      attestRes.SwitchRemote,
		"gpu_local_token":    attestRes.GPULocal.Token,
		"switch_local_token": attestRes.SwitchLocal.Token,
		"expected_nonce":     nonce,
	})
	if err != nil {
		return nil, errors.New("failed marshaling miner attest response")
	}

	req, err := http.NewRequest(
		"POST",
		fmt.Sprintf("%s/attest", c.Deps.Env.NVIDIA_ATTEST_ENDPOINT),
		bytes.NewBuffer(body),
	)
	if err != nil {
		log.Warnw("Failed to generate request to nvidia-attest", "error", err)
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Close = true
	resp, err := client.Do(req)
	if err != nil {
		log.Warnw("Failed sending request to nvidia-attest", "error", err)
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		log.Warnw(
			"Bad status code from nvidia-attest",
			"status",
			fmt.Sprintf("%d", resp.StatusCode),
		)
		return nil, errors.New("Bad status code from miner attest: " + resp.Status)
	}
	resBody, err := io.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		log.Warnw("Failed reading response body from nvidia-attest", "error", err)
		return nil, err
	}

	var attestResponse GPUAttestationResponse
	err = json.Unmarshal(resBody, &attestResponse)
	if err != nil {
		log.Debugw("Failed decoding json response from nvidia-attest", "error", err)
		return nil, err
	}

	if !attestResponse.GPUAttestationSuccess || !attestResponse.SwitchAttestationSuccess {
		log.Debugw("GPU or switch attestation failed",
			"gpu_success", attestResponse.GPUAttestationSuccess,
			"switch_success", attestResponse.SwitchAttestationSuccess)
		return nil, errors.New("GPU or switch attestation failed")
	}
	return &attestResponse, nil
}

func CheckCVMAttest(
	c *Core,
	client *http.Client,
	n *runtime.NeuronInfo,
	cvmIP string,
) ([]string, []string, error) {
	uid := fmt.Sprintf("%d", n.UID.Int64())
	Log := c.Deps.Log.With("uid", uid)
	h1 := strings.ReplaceAll(uuid.NewString(), "-", "")
	h2 := strings.ReplaceAll(uuid.NewString(), "-", "")
	nonce := h1 + h2
	attestRes, err := getCVMAttestFromNode(c, client, n, cvmIP, Log, nonce)
	if err != nil {
		return nil, nil, err
	}

	if !attestRes.GPULocal.AttestationResult {
		err = errors.New("local gpu attestation failed")
		Log.Debug(err.Error())
		return nil, nil, err
	}

	if !attestRes.GPULocal.Valid {
		err = errors.New("local gpu attestation invalid")
		Log.Debug(err.Error())
		return nil, nil, err
	}

	if !attestRes.GPURemote.AttestationResult {
		err = errors.New("remote gpu attestation failed")
		Log.Debug(err.Error())
		return nil, nil, err
	}

	if !attestRes.GPURemote.Valid {
		err = errors.New("remote gpu attestation invalid")
		Log.Debug(err.Error())
		return nil, nil, err
	}

	if !attestRes.SwitchLocal.AttestationResult {
		err = errors.New("local switch attestation failed")
		Log.Debug(err.Error())
		return nil, nil, err
	}

	if !attestRes.SwitchLocal.Valid {
		err = errors.New("local switch attestation invalid")
		Log.Debug(err.Error())
		return nil, nil, err
	}
	if !attestRes.SwitchRemote.AttestationResult {
		err = errors.New("remote switch attestation failed")
		Log.Debug(err.Error())
		return nil, nil, err
	}

	if !attestRes.SwitchRemote.Valid {
		err = errors.New("remote switch attestation invalid")
		Log.Debug(err.Error())
		return nil, nil, err
	}

	attestResponse, err := verifyAttestResponse(c, client, attestRes, nonce, Log)
	if err != nil {
		return nil, nil, err
	}

	// Extract GPU types from the claims
	var gpuTypes []string
	var snums []string
	if attestResponse.GPUClaims != nil {
		for _, claims := range attestResponse.GPUClaims {
			gpuTypes = append(gpuTypes, claims.GPUType)
			snums = append(snums, claims.GPUID)
		}
	}
	if attestResponse.SwitchClaims != nil {
		for _, claims := range attestResponse.SwitchClaims {
			snums = append(snums, claims.SwitchID)
		}
	}
	Log.Infow("GPU attestation successful",
		"gpu_types", fmt.Sprintf("%v", gpuTypes),
		"ip", cvmIP,
	)

	return gpuTypes, snums, nil
}
