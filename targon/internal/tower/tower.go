package tower

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"targon/internal/utils"

	"github.com/centrifuge/go-substrate-rpc-client/v4/signature"
	"github.com/subtrahend-labs/gobt/boilerplate"
	"go.uber.org/zap"
)

type Tower struct {
	client *http.Client
	log    *zap.SugaredLogger
	hotkey *signature.KeyringPair
	url    string
}

type AuctionDetails struct {
	TaoPrice float64 `json:"tao_price"`
	MaxBid   int     `json:"max_bid"`
}

func NewTower(client *http.Client, url string, hotkey *signature.KeyringPair, log *zap.SugaredLogger) *Tower {
	return &Tower{
		client: client,
		log:    log,
		hotkey: hotkey,
		url:    url,
	}
}

func (t *Tower) AuctionDetails() (*AuctionDetails, error) {
	res, err := t.client.Get(t.url + "/auction-details")
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
	var aucDetails AuctionDetails
	err = json.Unmarshal(body, &aucDetails)
	if err != nil {
		return nil, utils.Wrap("failed to unmarshal body", err)
	}
	return &aucDetails, nil
}

func (t *Tower) Check(
	ip string,
) bool {
	body, _ := json.Marshal(map[string]string{
		"ip_address": ip,
	})
	req, err := http.NewRequest("POST", t.url+"/check", bytes.NewBuffer(body))
	if err != nil {
		t.log.Debugw("Failed to generate request to tower", "error", err)
		return false
	}
	headers, err := boilerplate.GetEpistulaHeaders(
		*t.hotkey,
		"",
		body,
	)
	if err != nil {
		t.log.Debugw("Failed generating epistula headers", "error", err)
		return false
	}
	for key, value := range headers {
		req.Header.Set(key, value)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Close = true
	resp, err := t.client.Do(req)
	if err != nil {
		t.log.Debugw("Failed sending request to tower", "error", err)
		return false
	}
	defer func() {
		_ = resp.Body.Close()
	}()
	if resp.StatusCode != 200 {
		return false
	}
	resBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.log.Warnw("failed reading tower body", "error", err)
		return false
	}
	response := strings.TrimSpace(string(resBody))
	return response == "passed"
}
