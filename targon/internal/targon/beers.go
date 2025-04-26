package targon

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"targon/internal/setup"

	"github.com/subtrahend-labs/gobt/boilerplate"
)

type GPUData struct {
	Uid  string            `json:"uid"`
	Data map[string]string `json:"data"`
}

func sendGPUDataToBeers(c *Core, client *http.Client, data any) error {
	body, err := json.Marshal(data)
	if err != nil {
		return fmt.Errorf("failed to marshal GPU data: %w", err)
	}

	headers, err := boilerplate.GetEpistulaHeaders(c.Deps.Hotkey, "", body)
	if err != nil {
		return fmt.Errorf("failed generating epistula headers: %w", err)
	}

	req, err := http.NewRequest("POST", fmt.Sprintf("%s/mongo", setup.BEERS_URL), bytes.NewBuffer(body))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	for key, value := range headers {
		req.Header.Set(key, value)
	}
	req.Close = true

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send request to beers: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("beers endpoint returned non-200 status code: %d", resp.StatusCode)
	}

	return nil
}
