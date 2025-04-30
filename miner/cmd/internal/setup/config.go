package setup

import (
	"encoding/json"
	"io"
	"os"
)

type Config struct {
	Port          string   `json:"port,omitempty"`
	Nodes         []string `json:"nodes,omitempty"`
	HotkeySS58    string   `json:"hotkey_ss58,omitempty"`
	ChainEndpoint string   `json:"chain_endpoint,omitempty"`
	Debug         bool     `json:"debug,omitempty"`
	Netuid        *int     `json:"netuid,omitempty"`
	DiscordUrl    string   `json:"discord_url,omitempty"`
}

func LoadConfig() *Config {
	f, err := os.Open("./config.json")
	if err != nil {
		panic(err)
	}
	defer f.Close()
	var config Config
	bytes, err := io.ReadAll(f)
	if err != nil {
		panic(err)
	}
	json.Unmarshal(bytes, &config)

	if config.ChainEndpoint == "" {
		config.ChainEndpoint = "wss://entrypoint-finney.opentensor.ai:443"
	}
	defaultNetuid := 4
	if config.Netuid == nil {
		config.Netuid = &defaultNetuid
	}
	if config.Port == "" {
		panic("No port specified")
	}
	if config.HotkeySS58 == "" {
		panic("No hotkey specified")
	}
	return &config
}
