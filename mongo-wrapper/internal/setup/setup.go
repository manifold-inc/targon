package setup

import (
	"encoding/json"
	"fmt"
	"net"
	"os"
	"sync"
	"time"

	"github.com/centrifuge/go-substrate-rpc-client/v4/types"
	"github.com/subtrahend-labs/gobt/client"
	"github.com/subtrahend-labs/gobt/runtime"
	"github.com/subtrahend-labs/gobt/storage"
	"github.com/vedhavyas/go-subkey/v2"
	"go.mongodb.org/mongo-driver/v2/mongo"
	"go.uber.org/zap"
)

type Dependencies struct {
	Log      *zap.SugaredLogger
	Mongo    *mongo.Client
	mu       sync.Mutex
	IPs      IPs
	Client   *client.Client
	shutdown chan bool
}

type IPs struct {
	Validators map[string]string `json:"validators"`
	Targon     []string          `json:"targon"`
}

func Init() *Dependencies {
	cfg := zap.NewProductionConfig()
	cfg.Sampling = nil

	logger, err := cfg.Build()
	if err != nil {
		panic("Failed to get logger")
	}
	sugar := logger.Sugar()

	mongoClient, err := InitMongo()
	if err != nil {
		sugar.Fatalw("Failed to initialize MongoDB client", "error", err)
	}
	chainURL := os.Getenv("CHAIN_ENDPOINT")
	if chainURL == "" {
		chainURL = "wss://entrypoint-finney.opentensor.ai:443"
	}

	client, err := client.NewClient(chainURL)
	if err != nil {
		sugar.Fatalf("Error creating client: %s", err)
	}

	deps := &Dependencies{
		Log:      sugar,
		Client:   client,
		Mongo:    mongoClient,
		shutdown: make(chan bool),
	}
	if err := deps.UpdateIPs(); err != nil {
		deps.Log.Error("failed initializing ips: ", err)
		return nil
	}
	deps.Log.Info("updated ips successfully")
	go deps.WatchForIPs()
	return deps
}

func AccountIDToSS58(acc types.AccountID) string {
	recipientSS58 := subkey.SS58Encode(acc.ToBytes(), 42)
	return recipientSS58
}

func (d *Dependencies) Shutdown() {
	d.shutdown <- true
	<-d.shutdown
}

func (d *Dependencies) WatchForIPs() {
	timer := time.NewTicker(30 * time.Minute)
	for {
		select {
		case <-timer.C:
			d.Log.Info("fetching validator list")
			if err := d.UpdateIPs(); err != nil {
				d.Log.Error("failed initializing ips: ", err)
				continue
			}
			d.Log.Info("updated ips successfully")
		case <-d.shutdown:
			d.Log.Info("exited update loop")
			close(d.shutdown)
			return
		}
	}
}

func (d *Dependencies) UpdateIPs() error {
	d.Log.Info("Updating and registering nodes from config")

	// Grab vpermits
	permits, err := storage.GetValidatorPermits(d.Client, types.NewU16(uint16(4)), nil)
	if err != nil {
		d.Log.Errorw("Failed getting validator permits", "error", err)
		return err
	}

	// grab neurons
	blockHash, err := d.Client.Api.RPC.Chain.GetBlockHashLatest()
	if err != nil {
		return fmt.Errorf("failed getting blockhash for neurons: %s", err)
	}
	neurons, err := runtime.GetNeurons(d.Client, uint16(4), &blockHash)
	if err != nil {
		return fmt.Errorf("failed getting neurons: %s", err)
	}
	metagraph, err := runtime.GetMetagraph(d.Client, uint16(4), &blockHash)
	if err != nil {
		return fmt.Errorf("failed getting metagraph: %s", err)
	}

	// we need to make sure the map is reset
	d.mu.Lock()
	defer d.mu.Unlock()
	d.IPs = IPs{}
	file, err := os.ReadFile("/targon-ips.json")
	if err != nil {
		return fmt.Errorf("failed reading targon-ips: %s", err)
	}
	err = json.Unmarshal(file, &d.IPs)
	if err != nil {
		return fmt.Errorf("failed unmarshaling targon-ips: %s", err)
	}
	d.IPs.Validators = map[string]string{}

	for _, n := range neurons {
		// all entries in metagraph come indexed by uid
		stakeEntryInMetagraph := metagraph.TotalStake[int(n.UID.Int64())]
		// we need to use TotalStake because it includes the parent stake(s)
		n.Stake[0].Amount = stakeEntryInMetagraph
		if (*permits)[n.UID.Int64()] && n.Stake[0].Amount.Int64() > 1000 {
			if n.AxonInfo.IP.String() == "0" {
				continue
			}
			ss58 := AccountIDToSS58(n.Hotkey)
			var netip net.IP = n.AxonInfo.IP.Bytes()
			valiIP := fmt.Sprintf("http://%s:%d", netip, n.AxonInfo.Port)
			d.IPs.Validators[ss58] = valiIP
		}
	}

	d.Log.Info("ips Updated")
	return nil
}
