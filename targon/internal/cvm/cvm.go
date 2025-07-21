package cvm

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"time"

	"targon/internal/subtensor/utils"
	"targon/internal/targon"
	errutil "targon/internal/utils"

	"github.com/centrifuge/go-substrate-rpc-client/v4/signature"
	"github.com/subtrahend-labs/gobt/boilerplate"
	"github.com/subtrahend-labs/gobt/runtime"
)

func GetNodes(timeout_mult time.Duration, hotkey signature.KeyringPair, n *runtime.NeuronInfo) ([]*targon.MinerNode, error) {
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

type AttestBody struct {
	Nonce string `json:"nonce"`
}

type BusyError struct{}

func (b *BusyError) Error() string {
	return "node is busy"
}

func GetAttestFromNode(
	hotkey signature.KeyringPair,
	timeout_mult time.Duration,
	n *runtime.NeuronInfo,
	cvmIP string,
	nonce string,
) (*targon.AttestPayload, error) {
	client := &http.Client{Transport: &http.Transport{
		TLSHandshakeTimeout: 5 * time.Second * timeout_mult,
		MaxConnsPerHost:     1,
		DisableKeepAlives:   true,
		Dial: (&net.Dialer{
			Timeout: 15 * time.Second * timeout_mult,
		}).Dial,
	}, Timeout: 5 * time.Minute * timeout_mult}
	data := AttestBody{Nonce: nonce}
	body, _ := json.Marshal(data)
	req, err := http.NewRequest(
		"POST",
		fmt.Sprintf("http://%s:8080/api/v1/attest", cvmIP),
		bytes.NewBuffer(body),
	)
	if err != nil {
		return nil, errutil.Wrap("failed to generate request to miner", err)
	}
	headers, err := boilerplate.GetEpistulaHeaders(
		hotkey,
		utils.AccountIDToSS58(n.Hotkey),
		body,
	)
	if err != nil {
		return nil, errutil.Wrap("failed generating epistula headers", err)
	}

	req.Header.Set("Content-Type", "application/json")
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
	if resp.StatusCode == http.StatusServiceUnavailable {
		return nil, &BusyError{}
	}
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("bad status code from miner attest: %d: %s", resp.StatusCode, string(body))
	}
	resBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, errutil.Wrap("failed reading response", err)
	}
	var attestRes targon.AttestResponse
	err = json.Unmarshal(resBody, &attestRes)
	if err != nil {
		return nil, errutil.Wrap("failed unmarshaling response", err)
	}
	return &targon.AttestPayload{Attest: &attestRes}, nil
}

func verifyAttestResponse(
	attestEndpoint string,
	client *http.Client,
	attestRes *targon.AttestResponse,
	nonce string,
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
		fmt.Sprintf("%s/attest", attestEndpoint),
		bytes.NewBuffer(body),
	)
	if err != nil {
		return nil, errutil.Wrap("failed to generate request to nvidia-attest", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Close = true
	resp, err := client.Do(req)
	if err != nil {
		return nil, errutil.Wrap("failed sending request to nvidia-attest", err)
	}
	defer func() {
		_ = resp.Body.Close()
	}()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("bad status code from miner attest: %d", resp.StatusCode)
	}
	resBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, errutil.Wrap("failed reading response body from nvidia-attest", err)
	}

	var attestResponse targon.GPUAttestationResponse
	err = json.Unmarshal(resBody, &attestResponse)
	if err != nil {
		return nil, errutil.Wrap("failed decoding json response from nvidia-attest", err)
	}

	if !attestResponse.GPUAttestationSuccess || !attestResponse.SwitchAttestationSuccess {
		return nil, fmt.Errorf("GPU={%t} or switch attestation={%t} failed", attestResponse.GPUAttestationSuccess, attestResponse.SwitchAttestationSuccess)
	}
	return &attestResponse, nil
}

func CheckAttest(
	attestEndpoint string,
	client *http.Client,
	attestation *targon.AttestResponse,
	nonce string,
) ([]string, []string, error) {
	switch false {
	case attestation.GPULocal.AttestationResult:
		return nil, nil, errors.New("local gpu attestation failed")
	case attestation.GPULocal.Valid:
		return nil, nil, errors.New("local gpu attestation invalid")
	case attestation.GPURemote.AttestationResult:
		return nil, nil, errors.New("remote gpu attestation failed")
	case attestation.GPURemote.Valid:
		return nil, nil, errors.New("remote gpu attestation invalid")
	case attestation.SwitchLocal.AttestationResult:
		return nil, nil, errors.New("local switch attestation failed")
	case attestation.SwitchLocal.Valid:
		return nil, nil, errors.New("local switch attestation invalid")
	case attestation.SwitchRemote.AttestationResult:
		return nil, nil, errors.New("remote switch attestation failed")
	case attestation.SwitchRemote.Valid:
		return nil, nil, errors.New("remote switch attestation invalid")
	}

	attestResponse, err := verifyAttestResponse(attestEndpoint, client, attestation, nonce)
	if err != nil {
		return nil, nil, errutil.Wrap("couldnt verify attestation", err)
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
	return gpuTypes, ueids, nil
}
