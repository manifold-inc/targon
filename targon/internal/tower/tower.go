// Package tower
package tower

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/centrifuge/go-substrate-rpc-client/v4/signature"
	"github.com/manifold-inc/manifold-sdk/lib/utils"
	"go.uber.org/zap"
)

type Tower struct {
	client *http.Client
	log    *zap.SugaredLogger
	hotkey *signature.KeyringPair
	url    string
}

type Auctions struct {
	Auctions map[string]Auction `json:"auctions" bson:"auctions"`
	TaoPrice float64            `json:"tao_price" bson:"tao_price"`
}

type Auction struct {
	MaxBid         int `json:"max_bid" bson:"max_bid"`
	Emission       int `json:"emission" bson:"emission"`
	MinClusterSize int `json:"min_cluster_size" bson:"min_cluster_size"`
}

func NewTower(client *http.Client, url string, hotkey *signature.KeyringPair, log *zap.SugaredLogger) *Tower {
	return &Tower{
		client: client,
		log:    log,
		hotkey: hotkey,
		url:    url,
	}
}

func (t *Tower) AuctionDetails() (*Auctions, error) {
	// TODO @alan
	// This code will now run inside a vm that has an attestation server running
	// Get an attestation and send it to tower to get auctions
	req, err := http.NewRequest("POST", fmt.Sprintf("%s/api/v1/auctions", t.url), bytes.NewBuffer([]byte{}))
	if err != nil {
		return nil, utils.Wrap("failed to generate request to tower", err)
	}
	res, err := t.client.Do(req)
	if err != nil {
		return nil, utils.Wrap("failed to generate request to tower", err)
	}
	defer func() {
		_ = res.Body.Close()
	}()
	if res.StatusCode != 200 {
		return nil, utils.Wrap("non 200 status code", fmt.Errorf("%d", res.StatusCode))
	}
	body, err := io.ReadAll(res.Body)
	if err != nil {
		return nil, utils.Wrap("failed reading body", err)
	}
	var aucDetails Auctions
	err = json.Unmarshal(body, &aucDetails)
	if err != nil {
		return nil, utils.Wrap("failed to unmarshal body", err)
	}
	return &aucDetails, nil
}

func GetAttestation(log *zap.SugaredLogger, nonce string) (*AttestResponse, error) {
	tr := &http.Transport{
		TLSHandshakeTimeout: 5 * time.Second,
		MaxConnsPerHost:     1,
		DisableKeepAlives:   true,
	}
	client := &http.Client{Transport: tr, Timeout: 5 * time.Minute}
	data := AttestBody{Nonce: nonce}
	body, _ := json.Marshal(data)
	req, err := http.NewRequest(
		"POST",
		"http://localhost:8080/api/v1/evidence",
		bytes.NewBuffer(body),
	)
	if err != nil {
		log.Debugw("Failed to generate request to miner", "error", err)
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
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
	var attestRes AttestResponse
	err = json.Unmarshal(resBody, &attestRes)
	if err != nil {
		log.Debugw("Failed unmarshaling response", "error", err)
		return nil, err
	}
	return &attestRes, nil
}

type AttestResponse struct {
	Quote    string    `json:"quote"`
	UserData *UserData `json:"user_data"`
}

type UserData struct {
	// Added in attester
	GPUCards     *Cards        `json:"gpu_cards,omitempty"`
	CPUCards     *Cards        `json:"cpu_cards,omitempty"`
	NodeType     string        `json:"node_type"`
	NVCCResponse *NVCCResponse `json:"attestation,omitempty"`
	AuctionName  string        `json:"auction_name"`

	// Added in handler
	Nonce     string `json:"nonce"`
	CVMID     string `json:"cvm_id"`
	QuoteType string `json:"quote_type"`
}

type NVCCResponse struct {
	GPURemote struct {
		AttestationResult bool   `json:"attestation_result"`
		Token             string `json:"token"`
		Valid             bool   `json:"valid"`
	} `json:"gpu_remote"`
	SwitchRemote struct {
		AttestationResult bool   `json:"attestation_result"`
		Token             string `json:"token"`
		Valid             bool   `json:"valid"`
	} `json:"switch_remote"`
}

type Cards []string

type AttestBody struct {
	Nonce         string `json:"nonce"`
	AtlasDiskSize string `json:"atlas_disk_size"`
}
