package pyth

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"strconv"
)

const (
	PYTH_URL     = "https://hermes.pyth.network/v2/updates/price/latest"
	TAO_PRICE_ID = "0x410f41de235f2db824e562ea7ab2d3d3d4ff048316c61d629c0b93f58584e1af"
)

type PythData struct {
	Parsed []struct {
		Price struct {
			Expo  int    `json:"expo,omitempty"`
			Price string `json:"price,omitempty"`
		} `json:"price,omitempty"`
	} `json:"parsed,omitempty"`
}

func GetTaoPrice() (float64, error) {
	res, err := http.Get(fmt.Sprintf("%s?ids[]=%s", PYTH_URL, TAO_PRICE_ID))
	if err != nil {
		return 0, err
	}
	body, err := io.ReadAll(res.Body)
	if err != nil {
		return 0, err
	}
	var pythData PythData
	err = json.Unmarshal(body, &pythData)
	if err != nil {
		return 0, err
	}
	if len(pythData.Parsed) == 0 {
		return 0, errors.New("no price returned from pyth")
	}
	priceBeforeExpo, err := strconv.Atoi(pythData.Parsed[0].Price.Price)
	if err != nil {
		return 0, err
	}
	price := float64(priceBeforeExpo) * float64(math.Pow10(pythData.Parsed[0].Price.Expo))

	return price, nil
}
