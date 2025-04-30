package setup

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/centrifuge/go-substrate-rpc-client/v4/signature"
	"github.com/centrifuge/go-substrate-rpc-client/v4/types"
	"github.com/subtrahend-labs/gobt/client"
	"go.uber.org/zap"
)

type Dependencies struct {
	Log    *zap.SugaredLogger
	Client *client.Client
	Hotkey signature.KeyringPair
	Config Config
}

func Init() *Dependencies {
	c := LoadConfig()
	// Startup
	cfg := zap.NewProductionConfig()
	cfg.Sampling = nil
	logger, err := cfg.Build()
	if err != nil {
		panic("Failed to get logger")
	}
	sugar := logger.Sugar()

	if c.Debug {
		cfg := zap.NewDevelopmentConfig()
		cfg.Sampling = nil
		logger, err := cfg.Build()
		if err != nil {
			panic("Failed to get logger")
		}
		sugar = logger.Sugar()
	}

	client, err := client.NewClient(c.ChainEndpoint)
	if err != nil {
		sugar.Fatalf("Error creating client: %s", err)
	}
	kp, err := signature.KeyringPairFromSecret(c.HotkeyPhrase, client.Network)
	if err != nil {
		sugar.Fatalw("Failed creating keyring par", err)
	}
	c.HotkeyPhrase = ""
	sugar.Infof("Starting miner with config [hotkey phrase hidden] %+v", *c)
	return &Dependencies{
		Log:    sugar,
		Client: client,
		Config: *c,
		Hotkey: kp,
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
