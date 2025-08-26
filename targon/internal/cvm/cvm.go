// Package cvm
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

	"targon/internal/targon"
	errutil "targon/internal/utils"

	"github.com/centrifuge/go-substrate-rpc-client/v4/signature"
	"github.com/subtrahend-labs/gobt/boilerplate"
)

type AttestBody struct {
	Nonce string `json:"nonce"`
}

type Attester struct {
	client      *http.Client
	timeoutMult time.Duration
	Hotkey      signature.KeyringPair
	towerURL    string
}

// NewAttester Creates a new Attester
// client is used for hitting tower (reuseable)
// timeoutMult is used to scale all client timeouts
// hotkey is the senders hotkey for generating epistula headers
func NewAttester(
	timeoutMult time.Duration,
	hotkey signature.KeyringPair,
	towerURL string,
) *Attester {
	client := &http.Client{Transport: &http.Transport{
		TLSHandshakeTimeout: 5 * time.Second * timeoutMult,
		DisableKeepAlives:   true,
	}, Timeout: 1 * time.Minute * timeoutMult}
	return &Attester{client: client, towerURL: towerURL, timeoutMult: timeoutMult, Hotkey: hotkey}
}

func (a *Attester) GetAttestFromNode(
	minerHotkey string,
	cvmIP string,
	nonce string,
) (*targon.AttestationResponse, error) {
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
		fmt.Sprintf("http://%s:8080/api/v1/evidence", cvmIP),
		bytes.NewBuffer(body),
	)
	if err != nil {
		return nil, errutil.Wrap("failed to generate request to miner", err)
	}
	headers, err := boilerplate.GetEpistulaHeaders(
		a.Hotkey,
		minerHotkey,
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
		return nil, errors.New("server overloaded")
	}
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("bad status code from miner attest: %d: %s", resp.StatusCode, string(body))
	}
	resBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, errutil.Wrap("failed reading response", err)
	}
	var attestRes targon.AttestationResponse
	err = json.Unmarshal(resBody, &attestRes)
	if err != nil {
		return nil, errutil.Wrap("failed unmarshaling response", err)
	}
	return &attestRes, nil
}

func (a *Attester) VerifyAttestation(
	attestRes *targon.AttestationResponse,
	nonce string,
	ip string,
) (*targon.UserData, error) {
	// Validate Attestation
	body, err := json.Marshal(map[string]any{
		"attestation": attestRes,
		"ip_address":  ip,
	})
	if err != nil {
		return nil, errutil.Wrap("failed marshaling miner attest response", err)
	}

	req, err := http.NewRequest(
		"POST",
		fmt.Sprintf("%s/api/v1/verify-attestation", a.towerURL),
		bytes.NewBuffer(body),
	)
	if err != nil {
		return nil, errutil.Wrap("failed to generate request to tower", err)
	}
	headers, err := boilerplate.GetEpistulaHeaders(a.Hotkey, "", body)
	if err != nil {
		return nil, errutil.Wrap("failed creating epistula headers", err)
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Close = true
	resp, err := a.client.Do(req)
	if err != nil {
		return nil, errutil.Wrap("failed sending request to tower", err)
	}
	defer func() {
		_ = resp.Body.Close()
	}()
	resBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, errutil.Wrap("failed reading response body from tower", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, errutil.Wrap("bad status code from tower", errors.New(string(resBody)))
	}

	var attestResponse targon.GPUAttestationResponse
	err = json.Unmarshal(resBody, &attestResponse)
	if err != nil {
		return nil, errutil.Wrap("failed decoding json response from tower")
	}

	if !attestResponse.Valid {
		return nil, errutil.Wrap(
			"attestation invalid", attestResponse.Error,
		)
	}
	return &attestRes.UserData, nil
}

// GetNodes Gets a single miners cvm bids
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
				IP:    node,
				Price: 0,
			})
		}
	}
	return nodesv2, nil
}
