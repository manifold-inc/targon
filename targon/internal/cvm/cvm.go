package cvm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"time"

	"targon/internal/targon"
	errutil "targon/internal/utils"

	"github.com/centrifuge/go-substrate-rpc-client/v4/signature"
	"github.com/subtrahend-labs/gobt/boilerplate"
)

type AttestBody struct {
	Nonce string `json:"nonce"`
}

type AttestError struct {
	Msg         string
	ShouldRetry bool
}

func (b *AttestError) Error() string {
	return fmt.Sprintf("%s: retry %t", b.Msg, b.ShouldRetry)
}

type Attester struct {
	client         *http.Client
	timeoutMult    time.Duration
	Hotkey         signature.KeyringPair
	attestEndpoint string
}

// Creates a new Attester
// client is used for hitting nvidia-attestation verification endpoint (reuseable)
// timeoutMult is used to scale all client timeouts
// hotkey is the senders hotkey for generating epistula headers
func NewAttester(
	timeoutMult time.Duration,
	hotkey signature.KeyringPair,
	attestEndpoint string,
) *Attester {
	client := &http.Client{Transport: &http.Transport{
		TLSHandshakeTimeout: 5 * time.Second * timeoutMult,
		DisableKeepAlives:   true,
	}, Timeout: 3 * time.Minute * timeoutMult}
	return &Attester{client: client, timeoutMult: timeoutMult, Hotkey: hotkey, attestEndpoint: attestEndpoint}
}

func (a *Attester) GetAttestFromNode(
	minerHotkey string,
	cvmIP string,
	nonce string,
) (*targon.AttestResponse, error) {
	client := &http.Client{Transport: &http.Transport{
		TLSHandshakeTimeout: 5 * time.Second * a.timeoutMult,
		MaxConnsPerHost:     1,
		DisableKeepAlives:   true,
		Dial: (&net.Dialer{
			Timeout: 15 * time.Second * a.timeoutMult,
		}).Dial,
	}, Timeout: 5 * time.Minute * a.timeoutMult}
	data := AttestBody{Nonce: nonce}
	body, _ := json.Marshal(data)
	req, err := http.NewRequest(
		"POST",
		fmt.Sprintf("http://%s:8080/api/v1/attest", cvmIP),
		bytes.NewBuffer(body),
	)
	if err != nil {
		return nil, &AttestError{ShouldRetry: true, Msg: errutil.Wrap("failed to generate request to miner", err).Error()}
	}
	headers, err := boilerplate.GetEpistulaHeaders(
		a.Hotkey,
		minerHotkey,
		body,
	)
	if err != nil {
		return nil, &AttestError{ShouldRetry: true, Msg: errutil.Wrap("failed generating epistula headers", err).Error()}
	}
	req.Header.Set("Content-Type", "application/json")
	for key, value := range headers {
		req.Header.Set(key, value)
	}
	req.Close = true
	resp, err := client.Do(req)
	if err != nil {
		return nil, &AttestError{ShouldRetry: true, Msg: errutil.Wrap("failed sending request to miner", err).Error()}
	}
	defer func() {
		_ = resp.Body.Close()
	}()
	if resp.StatusCode == http.StatusServiceUnavailable {
		return nil, &AttestError{ShouldRetry: true, Msg: "server overloaded"}
	}
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, &AttestError{ShouldRetry: false, Msg: fmt.Sprintf("bad status code from miner attest: %d: %s", resp.StatusCode, string(body))}
	}
	resBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, &AttestError{ShouldRetry: true, Msg: errutil.Wrap("failed reading response", err).Error()}
	}
	var attestRes targon.AttestResponse
	err = json.Unmarshal(resBody, &attestRes)
	if err != nil {
		return nil, &AttestError{ShouldRetry: true, Msg: errutil.Wrap("failed unmarshaling response", err).Error()}
	}
	return &attestRes, nil
}

func (a *Attester) CheckAttest(
	attestation *targon.AttestResponse,
	nonce string,
) ([]string, []string, error) {
	switch false {
	case attestation.GPULocal.AttestationResult:
		return nil, nil, &AttestError{ShouldRetry: true, Msg: "local gpu attestation failed"}
	case attestation.GPULocal.Valid:
		return nil, nil, &AttestError{ShouldRetry: true, Msg: "local gpu attestation invalid"}
	case attestation.GPURemote.AttestationResult:
		return nil, nil, &AttestError{ShouldRetry: true, Msg: "remote gpu attestation failed"}
	case attestation.GPURemote.Valid:
		return nil, nil, &AttestError{ShouldRetry: true, Msg: "remote gpu attestation invalid"}
	case attestation.SwitchLocal.AttestationResult:
		return nil, nil, &AttestError{ShouldRetry: true, Msg: "local switch attestation failed"}
	case attestation.SwitchLocal.Valid:
		return nil, nil, &AttestError{ShouldRetry: true, Msg: "local switch attestation invalid"}
	case attestation.SwitchRemote.AttestationResult:
		return nil, nil, &AttestError{ShouldRetry: true, Msg: "remote switch attestation failed"}
	case attestation.SwitchRemote.Valid:
		return nil, nil, &AttestError{ShouldRetry: true, Msg: "remote switch attestation invalid"}
	}

	attestResponse, err := a.verifyAttestResponse(attestation, nonce)
	if err != nil {
		return nil, nil, &AttestError{ShouldRetry: err.ShouldRetry, Msg: errutil.Wrap("couldnt verify attestation", err).Error()}
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

func (a *Attester) verifyAttestResponse(
	attestRes *targon.AttestResponse,
	nonce string,
) (*targon.GPUAttestationResponse, *AttestError) {
	// Validate Attestation
	body, err := json.Marshal(map[string]any{
		"gpu_remote":     attestRes.GPURemote,
		"switch_remote":  attestRes.SwitchRemote,
		"gpu_local":      attestRes.GPULocal,
		"switch_local":   attestRes.SwitchLocal,
		"expected_nonce": nonce,
	})
	if err != nil {
		return nil, &AttestError{ShouldRetry: true, Msg: "failed marshaling miner attest response"}
	}

	req, err := http.NewRequest(
		"POST",
		fmt.Sprintf("%s/attest", a.attestEndpoint),
		bytes.NewBuffer(body),
	)
	if err != nil {
		return nil, &AttestError{ShouldRetry: true, Msg: errutil.Wrap("failed to generate request to nvidia-attest", err).Error()}
	}
	req.Header.Set("Content-Type", "application/json")
	req.Close = true
	resp, err := a.client.Do(req)
	if err != nil {
		return nil, &AttestError{ShouldRetry: true, Msg: errutil.Wrap("failed sending request to nvidia-attest", err).Error()}
	}
	defer func() {
		_ = resp.Body.Close()
	}()
	if resp.StatusCode != http.StatusOK {
		return nil, &AttestError{ShouldRetry: true, Msg: fmt.Sprintf("bad status code from miner attest: %d", resp.StatusCode)}
	}
	resBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, &AttestError{ShouldRetry: true, Msg: errutil.Wrap("failed reading response body from nvidia-attest", err).Error()}
	}

	var attestResponse targon.GPUAttestationResponse
	err = json.Unmarshal(resBody, &attestResponse)
	if err != nil {
		return nil, &AttestError{ShouldRetry: true, Msg: errutil.Wrap("failed decoding json response from nvidia-attest", err).Error()}
	}

	if !attestResponse.GPUAttestationSuccess || !attestResponse.SwitchAttestationSuccess {
		return nil, &AttestError{ShouldRetry: true, Msg: fmt.Sprintf("GPU={%t} or switch attestation={%t} failed", attestResponse.GPUAttestationSuccess, attestResponse.SwitchAttestationSuccess)}
	}
	return &attestResponse, nil
}

// Gets a single miners cvm bids
func (a *Attester) GetNodes(hotkey string, ip string) ([]*targon.MinerNode, error) {
	tr := &http.Transport{
		TLSHandshakeTimeout: 5 * time.Second * a.timeoutMult,
		MaxConnsPerHost:     1,
		DisableKeepAlives:   true,
	}
	client := &http.Client{Transport: tr, Timeout: 5 * time.Second * a.timeoutMult}
	req, err := http.NewRequest(
		"GET",
		fmt.Sprintf("http://%s/cvm", ip),
		nil,
	)
	if err != nil {
		return nil, errutil.Wrap("failed to generate request to miner", err)
	}
	headers, err := boilerplate.GetEpistulaHeaders(
		a.Hotkey,
		hotkey,
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
