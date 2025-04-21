package setup

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/centrifuge/go-substrate-rpc-client/v4/signature"
	"github.com/centrifuge/go-substrate-rpc-client/v4/types"
	"github.com/joho/godotenv"
	"github.com/subtrahend-labs/gobt/client"
	"go.uber.org/zap"
)

type Dependencies struct {
	Log    *zap.SugaredLogger
	Env    Env
	Client *client.Client
	Hotkey signature.KeyringPair
}
type Env struct {
	HOTKEY_PUBLIC_KEY      string
	HOTKEY_PRIVATE_KEY     string
	HOTKEY_SS58            string
	HOTKEY_PHRASE          string
	CHAIN_ENDPOINT         string
	NVIDIA_ATTEST_ENDPOINT string
	VERSION                types.U64
}

func GetEnv(key, fallback string) string {
	if value, ok := os.LookupEnv(key); ok {
		return value
	}
	return fallback
}

func GetEnvOrPanic(key string, logger *zap.SugaredLogger) string {
	if value, ok := os.LookupEnv(key); ok {
		return value
	}
	logger.Panicf("Could not find env key [%s]", key)
	return ""
}

func Init() *Dependencies {
	// Startup
	cfg := zap.NewProductionConfig()
	cfg.Sampling = nil
	logger, err := cfg.Build()
	if err != nil {
		panic("Failed to get logger")
	}
	sugar := logger.Sugar()

	// Env Variables
	err = godotenv.Load()
	if err != nil {
		sugar.Fatal("Error loading .env file")
	}
	HOTKEY_PHRASE := GetEnvOrPanic("HOTKEY_PHRASE", sugar)
	CHAIN_ENDPOINT := os.Getenv("CHAIN_ENDPOINT")
	NVIDIA_ATTEST_ENDPOINT := GetEnv("NVIDIA_ATTEST_ENDPOINT", "http://nvidia-attest")
	VERSION := GetEnvOrPanic("VERSION", sugar)
	parsedVer, err := ParseVersion(VERSION)
	if err != nil {
		sugar.Fatal(err)
	}

	client, err := client.NewClient(CHAIN_ENDPOINT)
	if err != nil {
		sugar.Fatalf("Error creating client: %s", err)
	}

	kp, err := signature.KeyringPairFromSecret(HOTKEY_PHRASE, network)
	if err != nil {
		sugar.Fatalw("Failed creating keyring par", err)
	}

	return &Dependencies{
		Log:    sugar,
		Client: client,
		Hotkey: kp,
		Env: Env{
			HOTKEY_PHRASE:          HOTKEY_PHRASE,
			CHAIN_ENDPOINT:         CHAIN_ENDPOINT,
			NVIDIA_ATTEST_ENDPOINT: NVIDIA_ATTEST_ENDPOINT,
			VERSION:                *parsedVer,
		},
	}
}

func ParseVersion(v string) (*types.U64, error) {
	parts := strings.Split(v, ".")
	if len(parts) != 3 {
		return nil, fmt.Errorf("not a valid version string: %v", v)
	}
	major, err := strconv.Atoi(parts[0])
	if err != nil {
		return nil, fmt.Errorf("not a valid version string: %v", v)
	}
	minor, err := strconv.Atoi(parts[1])
	if err != nil {
		return nil, fmt.Errorf("not a valid version string: %v", v)
	}
	patch, err := strconv.Atoi(parts[2])
	if err != nil {
		return nil, fmt.Errorf("not a valid version string: %v", v)
	}
	ver := (major * 100000) + (minor * 1000) + patch
	typedVer := types.NewU64(uint64(ver))
	return &typedVer, nil
}
