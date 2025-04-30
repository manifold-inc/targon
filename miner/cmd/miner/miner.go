package main

import (
	"fmt"
	"net/http"
	"sync"

	"miner/cmd/internal/setup"

	"github.com/centrifuge/go-substrate-rpc-client/v4/types"
	"github.com/labstack/echo/v4"
	"github.com/subtrahend-labs/gobt/boilerplate"
	"github.com/subtrahend-labs/gobt/runtime"
	"github.com/subtrahend-labs/gobt/storage"
)

type Core struct {
	Deps             *setup.Dependencies
	ValidatorPermits *[]types.Bool
	Neurons          map[string]runtime.NeuronInfo

	// Global core lock
	mu sync.Mutex
}

func CreateCore(d *setup.Dependencies) *Core {
	return &Core{
		Deps:    d,
		Neurons: map[string]runtime.NeuronInfo{},
	}
}

func UpdateCore(core *Core, h types.Header) {
	core.mu.Lock()
	defer core.mu.Unlock()

	// Grab vpermits
	res, err := storage.GetValidatorPermits(core.Deps.Client, types.NewU16(4), nil)
	if err != nil {
		core.Deps.Log.Errorw("Failed getting validator permits", "error", err)
		return
	}
	core.ValidatorPermits = res

	// grab neurons
	blockHash, err := core.Deps.Client.Api.RPC.Chain.GetBlockHash(uint64(h.Number))
	if err != nil {
		core.Deps.Log.Errorw("Failed getting blockhash for neurons", "error", err)
		return
	}
	neurons, err := runtime.GetNeurons(core.Deps.Client, uint16(4), &blockHash)
	if err != nil {
		core.Deps.Log.Errorw("Failed getting neurons", "error", err)
		return
	}

	// we need to make sure the map is reset
	core.Neurons = map[string]runtime.NeuronInfo{}
	for _, n := range neurons {
		core.Neurons[setup.AccountIDToSS58(n.Hotkey)] = n
	}
	core.Deps.Log.Info("Neurons Updated")
}

func main() {
	deps := setup.Init()
	deps.Log.Infof(
		"Starting validator with key [%s] on chain [%s]",
		deps.Hotkey.Address,
		deps.Config.ChainEndpoint,
	)

	core := CreateCore(deps)
	validator := boilerplate.NewChainSubscriber(*deps.Config.Netuid)
	deps.Log.Infof("Starting Miner on netuid [%d]", validator.NetUID)
	validator.AddBlockCallback(func(h types.Header) {
		uid := "Unknown"
		emi := "Unknown"
		ip := "Unknown"
		if len(core.Neurons) != 0 {
			n := core.Neurons[core.Deps.Hotkey.Address]
			uid = fmt.Sprintf("%d", n.UID.Int64())
			emi = fmt.Sprintf("%d", n.Emission.Int64())
			netip := n.AxonInfo.IP.Bytes()
			ip = fmt.Sprintf("http://%s:%d", netip, n.AxonInfo.Port)
		}
		core.Deps.Log.Infow(
			"New block",
			"block",
			fmt.Sprintf("%v", h.Number),
			"left_in_interval",
			fmt.Sprintf("%d", 360-(h.Number%360)),
			"uid",
			uid,
			"ip",
			ip,
			"Emission",
			emi,
		)
	})
	validator.AddBlockCallback(func(h types.Header) {
		if h.Number%360 != 1 && core.ValidatorPermits != nil {
			return
		}
		core.Deps.Log.Infow("Fetching validator list", "block", fmt.Sprintf("%v", h.Number))
		UpdateCore(core, h)
	})
	validator.SetMainFunc(func(i <-chan bool, o chan<- bool) {
		e := echo.New()
		e.GET("/cvm", func(c echo.Context) error {
			sig := c.Request().Header.Get("Epistula-Request-Signature")
			timestamp := c.Request().Header.Get("Epistula-Timestamp")
			uuid := c.Request().Header.Get("Epistula-Uuid")
			signed_for := c.Request().Header.Get("Epistula-Signed-For")
			signed_by := c.Request().Header.Get("Epistula-Signed-By")
			err := boilerplate.VerifyEpistulaHeaders(
				core.Deps.Hotkey.Address,
				sig,
				[]byte{},
				timestamp,
				uuid,
				signed_for,
				signed_by,
			)
			// Failed signature
			if err != nil {
				return c.String(http.StatusForbidden, "Invalid Signature")
			}
			core.mu.Lock()
			defer core.mu.Unlock()

			// VPermit array is not ready
			if core.ValidatorPermits == nil {
				return c.String(http.StatusInternalServerError, "Still starting up...")
			}

			// Neuron hotkey not found
			neuron, ok := core.Neurons[signed_by]
			if !ok {
				return c.String(http.StatusForbidden, "Not valid signer")
			}

			// No vpermit found for validator
			if !(*core.ValidatorPermits)[int(neuron.UID.Int64())] {
				return c.String(http.StatusForbidden, "No VPermit")
			}
			deps.Log.Infof("Getting request from [%s]", signed_by)
			return c.JSON(http.StatusOK, core.Deps.Config.Nodes)
		})
		e.GET("/", func(c echo.Context) error {
			return c.String(http.StatusOK, "PONG")
		})
		e.Logger.Fatal(e.Start(":1323"))
	})

	validator.SetOnSubscriptionCreationError(func(e error) {
		deps.Log.Infow("Failed to connect to chain", "error", e)
		panic(e)
	})
	validator.SetOnSubscriptionError(func(e error) {
		deps.Log.Infow("Subscription Error", "error", e)
	})
	h, err := core.Deps.Client.Api.RPC.Chain.GetHeaderLatest()
	if err != nil {
		deps.Log.Fatal("Failed to get initial header")
	}
	UpdateCore(core, *h)
	if !CheckAlreadyRegistered(core) {
		core.Deps.Log.Info("Setting miner info, differs from config")
		err = setup.ServeMiner(deps)
		if err != nil {
			deps.Log.Errorw("Failed serving extrinsic", "error", err)
		}
	} else {
		core.Deps.Log.Info("Skipping set miner info, already set to config settings")
	}
	validator.Start(deps.Client)
}

func CheckAlreadyRegistered(core *Core) bool {
	n := core.Neurons[core.Deps.Hotkey.Address]
	netip := n.AxonInfo.IP.Bytes()
	currentIp := fmt.Sprintf("http://%s:%d", netip, n.AxonInfo.Port)
	configIp := fmt.Sprintf("http://%s:%d", core.Deps.Config.Ip, core.Deps.Config.Port)
	return currentIp == configIp
}
