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

func main() {
	deps := setup.Init()
	deps.Log.Infof(
		"Starting validator with key [%s] on chain [%s] version [%d]",
		deps.Hotkey.Address,
		deps.Env.CHAIN_ENDPOINT,
	)

	core := CreateCore(deps)
	validator := boilerplate.NewChainSubscriber(deps.Env.NETUID)
	deps.Log.Infof("Starting Miner on netuid [%d]", validator.NetUID)
	validator.AddBlockCallback(func(h types.Header) {
		core.Deps.Log.Infow(
			"New block",
			"block",
			fmt.Sprintf("%v", h.Number),
			"left_in_interval",
			fmt.Sprintf("%d", 360-(h.Number%360)),
		)
	})
	validator.AddBlockCallback(func(h types.Header) {
		if h.Number%360 != 1 && core.ValidatorPermits != nil {
			return
		}
		core.Deps.Log.Infow("Fetching validator list", "block", fmt.Sprintf("%v", h.Number))
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
				core.Deps.Env.HOTKEY_SS58,
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
	validator.Start(deps.Client)
}
