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
	Config Config
}
type Env struct {
	HOTKEY_SS58    string
	MINER_PORT     string
	CHAIN_ENDPOINT string
	DEBUG          bool
	NETUID         int
	DISCORD_URL    string
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
		sugar.Fatalw("Error loading .env file", err)
	}
	HOTKEY_SS58 := GetEnvOrPanic("HOTKEY_SS58", sugar)
	MINER_PORT := GetEnvOrPanic("MINER_PORT", sugar)
	CHAIN_ENDPOINT := GetEnv("CHAIN_ENDPOINT", "wss://entrypoint-finney.opentensor.ai:443")
	DISCORD_URL := GetEnv("DISCORD_URL", "")
	DEBUG := GetEnv("DEBUG", "0")
	netuid, err := strconv.Atoi(GetEnv("NETUID", "4"))
	if err != nil {
		sugar.Fatalw("Invalid netuid", "error", err)
	}
	debug := DEBUG == "1"
	if debug {
		cfg := zap.NewDevelopmentConfig()
		cfg.Sampling = nil
		logger, err := cfg.Build()
		if err != nil {
			panic("Failed to get logger")
		}
		sugar = logger.Sugar()
	}

	client, err := client.NewClient(CHAIN_ENDPOINT)
	if err != nil {
		sugar.Fatalf("Error creating client: %s", err)
	}

	return &Dependencies{
		Log:    sugar,
		Client: client,
		Env: Env{
			HOTKEY_SS58:    HOTKEY_SS58,
			DEBUG:          debug,
			CHAIN_ENDPOINT: CHAIN_ENDPOINT,
			NETUID:         netuid,
			DISCORD_URL:    DISCORD_URL,
			MINER_PORT:     MINER_PORT,
		},
		Config: *LoadConfig(),
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
