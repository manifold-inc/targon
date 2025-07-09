package cvm

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
	"targon/internal/targon"
	errutil "targon/internal/utils"

	"github.com/subtrahend-labs/gobt/boilerplate"
	"github.com/subtrahend-labs/gobt/runtime"
	"go.uber.org/zap"
)

func GetNodes(c *targon.Core, client *http.Client, n *runtime.NeuronInfo) ([]*targon.MinerNode, error) {
	if n.AxonInfo.IP.String() == "0" {
		err := errors.New("inactive miner")
		return nil, err
	}
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
		c.Deps.Hotkey,
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
				Price: 240,
			})
		}
	}

	// Max price is max bid, min price is 1
	for _, v := range nodesv2 {
		// TODO swap these lines to enable auctions
		v.Price = 300
		// v.Price = max(min(v.Price, c.MaxBid), 1)
	}
	return nodesv2, nil
}

func CheckHealth(c *targon.Core, client *http.Client, n *runtime.NeuronInfo, cvmIP string) bool {
	uid := fmt.Sprintf("%d", n.UID.Int64())
	Log := c.Deps.Log.With("uid", uid)
	cvmIP = strings.TrimPrefix(cvmIP, "http://")
	cvmIP = strings.TrimSuffix(cvmIP, ":8080")
	req, err := http.NewRequest("GET", fmt.Sprintf("http://%s:8080/health", cvmIP), nil)
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
	defer func() {
		_ = resp.Body.Close()
	}()
	return resp.StatusCode == http.StatusOK
}

type AttestBody struct {
	Nonce string `json:"nonce"`
}

func GetAttestFromNode(
	log *zap.SugaredLogger,
	c *targon.Core,
	client *http.Client,
	n *runtime.NeuronInfo,
	cvmIP string,
	nonce string,
) (*targon.AttestPayload, error) {
	data := AttestBody{Nonce: nonce}
	body, _ := json.Marshal(data)
	req, err := http.NewRequest(
		"POST",
		fmt.Sprintf("http://%s:8080/api/v1/attest", cvmIP),
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
	defer func() {
		_ = resp.Body.Close()
	}()
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
	icon := resp.Header.Get("X-Targon-ICON")
	if len(icon) == 0 {
		icon = "false"
	}
	var attestRes targon.AttestResponse
	err = json.Unmarshal(resBody, &attestRes)
	if err != nil {
		log.Debugw("Failed unmarshaling response", "error", err)
		return nil, err
	}
	return &targon.AttestPayload{Attest: &attestRes, ICON: icon}, nil
}

func verifyAttestResponse(
	c *targon.Core,
	client *http.Client,
	attestRes *targon.AttestResponse,
	nonce string,
	log *zap.SugaredLogger,
) (*targon.GPUAttestationResponse, error) {
	// Validate Attestation
	body, err := json.Marshal(map[string]any{
		"gpu_remote":     attestRes.GPURemote,
		"switch_remote":  attestRes.SwitchRemote,
		"gpu_local":      attestRes.GPULocal,
		"switch_local":   attestRes.SwitchLocal,
		"expected_nonce": nonce,
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
	defer func() {
		_ = resp.Body.Close()
	}()
	if resp.StatusCode != http.StatusOK {
		log.Warnw(
			"Bad status code from nvidia-attest",
			"status",
			fmt.Sprintf("%d", resp.StatusCode),
		)
		return nil, errors.New("Bad status code from miner attest: " + resp.Status)
	}
	resBody, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Warnw("Failed reading response body from nvidia-attest", "error", err)
		return nil, err
	}

	var attestResponse targon.GPUAttestationResponse
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

func CheckAttest(
	log *zap.SugaredLogger,
	c *targon.Core,
	client *http.Client,
	attestation *targon.AttestResponse,
	nonce string,
) ([]string, []string, error) {
	var err error
	if !attestation.GPULocal.AttestationResult {
		err = errors.New("local gpu attestation failed")
		log.Debug(err.Error())
		return nil, nil, err
	}

	if !attestation.GPULocal.Valid {
		err = errors.New("local gpu attestation invalid")
		log.Debug(err.Error())
		return nil, nil, err
	}

	if !attestation.GPURemote.AttestationResult {
		err = errors.New("remote gpu attestation failed")
		log.Debug(err.Error())
		return nil, nil, err
	}

	if !attestation.GPURemote.Valid {
		err = errors.New("remote gpu attestation invalid")
		log.Debug(err.Error())
		return nil, nil, err
	}

	if !attestation.SwitchLocal.AttestationResult {
		err = errors.New("local switch attestation failed")
		log.Debug(err.Error())
		return nil, nil, err
	}

	if !attestation.SwitchLocal.Valid {
		err = errors.New("local switch attestation invalid")
		log.Debug(err.Error())
		return nil, nil, err
	}
	if !attestation.SwitchRemote.AttestationResult {
		err = errors.New("remote switch attestation failed")
		log.Debug(err.Error())
		return nil, nil, err
	}

	if !attestation.SwitchRemote.Valid {
		err = errors.New("remote switch attestation invalid")
		log.Debug(err.Error())
		return nil, nil, err
	}

	attestResponse, err := verifyAttestResponse(c, client, attestation, nonce, log)
	if err != nil {
		return nil, nil, err
	}

	// Extract GPU types from the claims
	var gpuTypes []string
	if attestResponse.GPUClaims != nil {
		for _, claims := range attestResponse.GPUClaims {
			gpuTypes = append(gpuTypes, claims.GPUType)
		}
	}
	ueids := attestResponse.GPUIds
	if attestResponse.SwitchClaims != nil {
		for _, claims := range attestResponse.SwitchClaims {
			ueids = append(ueids, claims.SwitchID)
		}
	}
	log.Infow("GPU attestation successful",
		"gpu_types", fmt.Sprintf("%v", gpuTypes),
	)

	return gpuTypes, ueids, nil
}
