package setup

import (
	"encoding/json"
	"io"
	"os"
)

type Config struct {
	Port  string   `json:"port"`
	Nodes []string `json:"nodes"`
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

	return &config
}
