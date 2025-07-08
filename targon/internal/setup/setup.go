package setup

import (
	"fmt"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"targon/internal/discord"
	"targon/internal/tower"
	"targon/internal/utils"

	"github.com/centrifuge/go-substrate-rpc-client/v4/signature"
	"github.com/centrifuge/go-substrate-rpc-client/v4/types"
	"github.com/joho/godotenv"
	"github.com/subtrahend-labs/gobt/client"
	"go.mongodb.org/mongo-driver/v2/mongo"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

type Dependencies struct {
	Log    *zap.SugaredLogger
	Env    Env
	Client *client.Client
	Hotkey signature.KeyringPair
	Mongo  *mongo.Client
	Tower  *tower.Tower
}
type Env struct {
	HOTKEY_PHRASE          string
	ATTEST_RATE            float64
	CHAIN_ENDPOINT         string
	NVIDIA_ATTEST_ENDPOINT string
	VERSION                types.U64
	DEBUG                  bool
	NETUID                 int
	DISCORD_URL            string
	TIMEOUT_MULT           time.Duration
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

func Init(opts ...any) *Dependencies {
	var level *zapcore.Level
	if len(opts) != 0 {
		l := opts[0].(zapcore.Level)
		level = &l
	}
	// Startup
	cfg := zap.NewProductionConfig()
	cfg.Sampling = nil
	if level != nil {
		cfg.Level.SetLevel(*level)
	}
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
	DISCORD_URL := GetEnv("DISCORD_URL", "")
	HOTKEY_PHRASE := GetEnvOrPanic("HOTKEY_PHRASE", sugar)
	CHAIN_ENDPOINT := GetEnv("CHAIN_ENDPOINT", "wss://entrypoint-finney.opentensor.ai:443")
	NVIDIA_ATTEST_ENDPOINT := GetEnv("NVIDIA_ATTEST_ENDPOINT", "http://nvidia-attest")
	VERSION := GetEnvOrPanic("VERSION", sugar)
	DEBUG := GetEnv("DEBUG", "0")
	TOWER_URL := GetEnv("TOWER_URL", "https://tower.targon.com")
	TIMEOUT_MULT_STR := GetEnv("TIMEOUT_MULT", "1")
	TIMEOUT_MULT, err := strconv.Atoi(TIMEOUT_MULT_STR)
	if err != nil {
		sugar.Error("Failed converting env variable TIMEOUT_MULT to int")
		TIMEOUT_MULT = 1
	}
	sugar.Infof("Running with TIMEOUT_MULT=%d", time.Duration(TIMEOUT_MULT))

	ATTEST_RATE_STR := GetEnv("ATTEST_RATE", ".95")
	ATTEST_RATE, err := strconv.ParseFloat(ATTEST_RATE_STR, 64)
	if err != nil {
		sugar.Error("Failed converting env ATTEST_RATE to float")
		ATTEST_RATE = .95
		TIMEOUT_MULT = 1
	}
	ATTEST_RATE = min(ATTEST_RATE, .95)
	sugar.Infof("Running with ATTEST_RATE=%f", ATTEST_RATE)

	mongoClient, err := InitMongo()
	if err != nil {
		sugar.Warn(utils.Wrap("mongo error", err))
	}

	netuid, err := strconv.Atoi(GetEnv("NETUID", "4"))
	if err != nil {
		sugar.Fatalw("Invalid netuid", "error", err)
	}
	parsedVer, err := ParseVersion(VERSION)
	if err != nil {
		sugar.Fatal(err)
	}
	debug := DEBUG == "1"
	if debug {
		cfg := zap.NewDevelopmentConfig()
		cfg.Sampling = nil
		if level != nil {
			cfg.Level.SetLevel(*level)
		}
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

	kp, err := signature.KeyringPairFromSecret(HOTKEY_PHRASE, client.Network)
	if err != nil {
		sugar.Fatalw("Failed creating keyring par", err)
	}
	sugar = sugar.WithOptions(zap.Hooks(
		func(e zapcore.Entry) error {
			if e.Level != zap.ErrorLevel {
				return nil
			}
			go func() {
				color := "15548997"
				title := "Validator Error"
				desc := fmt.Sprintf("%s\n\n%s", e.Message, e.Stack)
				uname := "Validator Logs"
				msg := discord.Message{
					Username: &uname,
					Embeds: &[]discord.Embed{{
						Title:       &title,
						Description: &desc,
						Color:       &color,
					}},
				}
				_ = discord.SendDiscordMessage(DISCORD_URL, msg)
			}()

			return nil
		},
	))

	towerClient := &http.Client{Transport: &http.Transport{
		TLSHandshakeTimeout: 5 * time.Second * time.Duration(TIMEOUT_MULT),
		DisableKeepAlives:   true,
	}, Timeout: 1 * time.Minute * time.Duration(TIMEOUT_MULT)}
	t := tower.NewTower(towerClient, TOWER_URL, &kp, sugar)

	return &Dependencies{
		Log:    sugar,
		Client: client,
		Hotkey: kp,
		Mongo:  mongoClient,
		Tower:  t,
		Env: Env{
			DEBUG:                  debug,
			CHAIN_ENDPOINT:         CHAIN_ENDPOINT,
			NVIDIA_ATTEST_ENDPOINT: NVIDIA_ATTEST_ENDPOINT,
			VERSION:                *parsedVer,
			NETUID:                 netuid,
			DISCORD_URL:            DISCORD_URL,
			TIMEOUT_MULT:           time.Duration(TIMEOUT_MULT),
			ATTEST_RATE:            ATTEST_RATE,
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
	ver := (major * 100000) + (minor * 100) + patch
	typedVer := types.NewU64(uint64(ver))
	return &typedVer, nil
}
