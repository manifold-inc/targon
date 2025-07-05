package setup

import (
	"encoding/json"
	"io"
	"os"
)

type Config struct {
	Port          int        `json:"port,omitempty"`
	Nodes         []NodeItem `json:"nodes,omitempty"`
	HotkeyPhrase  string     `json:"hotkey_phrase,omitempty"`
	ChainEndpoint string     `json:"chain_endpoint,omitempty"`
	Debug         bool       `json:"debug,omitempty"`
	Netuid        *int       `json:"netuid,omitempty"`
	DiscordUrl    string     `json:"discord_url,omitempty"`
	Ip            string     `json:"ip,omitempty"`
	MinStake      int        `json:"min_stake"`
}
type NodeItem struct {
	Ip    string `json:"ip"`
	Price int    `json:"price"`
}

func LoadConfig() *Config {
	f, err := os.Open("./config.json")
	if err != nil {
		panic(err)
	}
	defer func() {
		_ = f.Close()
	}()
	var config Config
	bytes, err := io.ReadAll(f)
	if err != nil {
		panic(err)
	}
	err = json.Unmarshal(bytes, &config)
	if err != nil {
		panic("Failed reading config")
	}

	if config.ChainEndpoint == "" {
		config.ChainEndpoint = "wss://entrypoint-finney.opentensor.ai:443"
	}
	defaultNetuid := 4
	if config.Netuid == nil {
		config.Netuid = &defaultNetuid
	}
	if config.Port == 0 {
		config.Port = 7777
	}
	if config.HotkeyPhrase == "" {
		panic("No hotkey specified")
	}
	if config.Ip == "" {
		panic("No ip specified")
	}
	if config.MinStake == 0 {
		config.MinStake = 1000
	}
	return &config
}
