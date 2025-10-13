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

	"github.com/centrifuge/go-substrate-rpc-client/v4/signature"
	"github.com/centrifuge/go-substrate-rpc-client/v4/types"
	"github.com/joho/godotenv"
	"github.com/manifold-inc/manifold-sdk/lib/utils"
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
	TowerURL             string
	HotkeyPhrase         string
	AttestRate           float64
	ChainEndpoint        string
	NvidiaAttestEndpoint string
	Version              types.U64
	Debug                bool
	Netuid               int
	DiscordURL           string
	TimeoutMult          time.Duration
	ValiIP               string
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
	DiscordURL := GetEnv("DISCORD_URL", "")
	HotkeyPhrase := GetEnvOrPanic("HOTKEY_PHRASE", sugar)
	ChainEndpoint := GetEnv("CHAIN_ENDPOINT", "wss://entrypoint-finney.opentensor.ai:443")
	NvidiaAttestEndpoint := GetEnv("NVIDIA_ATTEST_ENDPOINT", "http://nvidia-attest")
	Version := GetEnvOrPanic("VERSION", sugar)
	ValiIP := GetEnvOrPanic("VALIDATOR_IP", sugar)
	Debug := GetEnv("DEBUG", "0")
	TowerURL := GetEnv("TOWER_URL", "https://tower.targon.com")
	TimeoutMultStr := GetEnv("TIMEOUT_MULT", "1")
	TimeoutMult, err := strconv.Atoi(TimeoutMultStr)
	if err != nil {
		sugar.Error("Failed converting env variable TIMEOUT_MULT to int")
		TimeoutMult = 1
	}
	sugar.Infof("Running with TIMEOUT_MULT=%d", time.Duration(TimeoutMult))

	AttestRateStr := GetEnv("ATTEST_RATE", ".95")
	AttestRate, err := strconv.ParseFloat(AttestRateStr, 64)
	if err != nil {
		sugar.Error("Failed converting env ATTEST_RATE to float")
		AttestRate = .95
		TimeoutMult = 1
	}
	AttestRate = min(AttestRate, .95)
	sugar.Infof("Running with ATTEST_RATE=%f", AttestRate)

	mongoClient, err := InitMongo()
	if err != nil {
		sugar.Fatal(utils.Wrap("failed connecting to mongo error", err))
	}

	netuid, err := strconv.Atoi(GetEnv("NETUID", "4"))
	if err != nil {
		sugar.Fatalw("Invalid netuid", "error", err)
	}
	parsedVer, err := ParseVersion(Version)
	if err != nil {
		sugar.Fatal(err)
	}
	debug := Debug == "1"
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

	client, err := client.NewClient(ChainEndpoint)
	if err != nil {
		sugar.Fatalf("Error creating client: %s", err)
	}

	kp, err := signature.KeyringPairFromSecret(HotkeyPhrase, client.Network)
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
				_ = discord.SendDiscordMessage(DiscordURL, msg)
			}()

			return nil
		},
	))

	towerClient := &http.Client{Transport: &http.Transport{
		TLSHandshakeTimeout: 5 * time.Second * time.Duration(TimeoutMult),
		DisableKeepAlives:   true,
	}, Timeout: 1 * time.Minute * time.Duration(TimeoutMult)}
	t := tower.NewTower(towerClient, TowerURL, &kp, sugar)

	return &Dependencies{
		Log:    sugar,
		Client: client,
		Hotkey: kp,
		Mongo:  mongoClient,
		Tower:  t,
		Env: Env{
			TowerURL:             TowerURL,
			Debug:                debug,
			ChainEndpoint:        ChainEndpoint,
			NvidiaAttestEndpoint: NvidiaAttestEndpoint,
			Version:              *parsedVer,
			Netuid:               netuid,
			DiscordURL:           DiscordURL,
			TimeoutMult:          time.Duration(TimeoutMult),
			AttestRate:           AttestRate,
			ValiIP:               ValiIP,
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
	ver := (major * 10000) + (minor * 100) + patch
	typedVer := types.NewU64(uint64(ver))
	return &typedVer, nil
}
