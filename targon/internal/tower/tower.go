package tower

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"targon/internal/utils"

	"github.com/centrifuge/go-substrate-rpc-client/v4/signature"
	"go.uber.org/zap"
)

type Tower struct {
	client *http.Client
	log    *zap.SugaredLogger
	hotkey *signature.KeyringPair
	url    string
}

type Auctions struct {
	TaoPrice float64            `json,bson:"tao_price"`
	Auctions map[string]Auction `json,bson:"auctions"`
}

type Auction struct {
	MaxBid         int    `json,bson:"max_bid" yaml:"MaxBid"`
	Emission       int    `json,bson:"emission" yaml:"Emission"`
	MinClusterSize int    `json,bson:"min_cluster_size" yaml:"MinClusterSize"`
	NodeType       string `json,bson:"node_type" yaml:"NodeType"`
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
	res, err := t.client.Get(t.url + "/api/v1/auctions")
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
