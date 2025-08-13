package main

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"miner/internal/setup"

	"github.com/centrifuge/go-substrate-rpc-client/v4/types"
	"github.com/labstack/echo/v4"
	"github.com/subtrahend-labs/gobt/boilerplate"
	"github.com/subtrahend-labs/gobt/runtime"
	"github.com/subtrahend-labs/gobt/storage"
)

func Wrap(msg string, errs ...error) error {
	fullerr := msg
	for _, err := range errs {
		if err == nil {
			continue
		}
		fullerr = fmt.Sprintf("%s: %s", fullerr, err)
	}
	return errors.New(fullerr)
}

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

func UpdateCore(core *Core) error {
	core.mu.Lock()
	defer core.mu.Unlock()

	core.Deps.Log.Info("Updating and registering nodes from config")
	core.Deps.NodeMu.Lock()
	core.Deps.Config = setup.LoadConfig()
	core.Deps.NodeMu.Unlock()

	// Grab vpermits
	res, err := storage.GetValidatorPermits(core.Deps.Client, types.NewU16(uint16(*core.Deps.Config.Netuid)), nil)
	if err != nil {
		core.Deps.Log.Errorw("Failed getting validator permits", "error", err)
		return err
	}
	core.ValidatorPermits = res

	// grab neurons
	blockHash, err := core.Deps.Client.Api.RPC.Chain.GetBlockHashLatest()
	if err != nil {
		return Wrap("failed getting blockhash for neurons", err)
	}
	neurons, err := runtime.GetNeurons(core.Deps.Client, uint16(*core.Deps.Config.Netuid), &blockHash)
	if err != nil {
		return Wrap("Failed getting neurons", err)
	}

	// we need to make sure the map is reset
	core.Neurons = map[string]runtime.NeuronInfo{}
	for _, n := range neurons {
		core.Neurons[setup.AccountIDToSS58(n.Hotkey)] = n
	}
	core.Deps.Log.Info("Neurons Updated")
	return nil
}

func main() {
	deps := setup.Init()
	deps.Log.Infof(
		"Starting miner with key [%s] on chain [%s]",
		deps.Hotkey.Address,
		deps.Config.ChainEndpoint,
	)

	core := CreateCore(deps)
	deps.Log.Infof("Starting Miner on netuid [%d]", *deps.Config.Netuid)

	// Update validator and neurons list
	shutdown := make(chan bool, 1)
	go func() {
		timer := time.NewTicker(30 * time.Minute)
		for {
			select {
			case <-timer.C:
				core.Deps.Log.Info("fetching validator list")
				if err := UpdateCore(core); err != nil {
					core.Deps.Log.Error("failed updating core: ", err)
					continue
				}
				core.Deps.Log.Info("updated core successfully")
			case <-shutdown:
				core.Deps.Log.Info("exited update loop")
				return
			}
		}
	}()

	e := echo.New()
	e.GET("/cvm", func(c echo.Context) error {
		deps.Log.Infof("Getting request from [%s]", c.RealIP())
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
			deps.Log.Warnf("Failed signature with error: %s", err)
			return c.String(http.StatusForbidden, "Invalid Signature")
		}
		core.mu.Lock()
		defer core.mu.Unlock()

		// VPermit array is not ready
		if core.ValidatorPermits == nil {
			deps.Log.Warn("Validator permits is nil")
			return c.String(http.StatusInternalServerError, "Still starting up...")
		}

		// Neuron hotkey not found
		neuron, ok := core.Neurons[signed_by]
		if !ok {
			deps.Log.Warnf("Signed_by not found in neurons: %s", signed_by)
			return c.String(http.StatusForbidden, "Not valid signer")
		}

		// No vpermit found for validator
		if !(*core.ValidatorPermits)[int(neuron.UID.Int64())] {
			deps.Log.Warnf("No vpermit for %s", signed_by)
			return c.String(http.StatusForbidden, "No VPermit")
		}

		stake := neuron.Stake[0].Amount.Int64()
		stakeInAlpha := stake / 1e9
		// Check if stake is below min stake, default 1000
		if stakeInAlpha < int64(deps.Config.MinStake) {
			deps.Log.Warnf("Stake is too low: %dt", stakeInAlpha)
			return c.String(http.StatusForbidden, "Stake too low")
		}

		deps.Log.Infof("Responding to request from request from [%s]", signed_by)

		return c.JSON(http.StatusOK, core.Deps.Config.Nodes)
	})
	e.GET("/", func(c echo.Context) error {
		return c.String(http.StatusOK, "PONG")
	})

	err := UpdateCore(core)
	if err != nil {
		core.Deps.Log.Error(Wrap("failed updating core", err))
		close(shutdown)
		return
	}
	if !CheckAlreadyRegistered(core) {
		core.Deps.Log.Info("Setting miner info, differs from config")
		err := setup.ServeMiner(deps)
		if err != nil {
			deps.Log.Errorw("Failed serving extrinsic", "error", err)
		}
	} else {
		core.Deps.Log.Info("Skipping set miner info, already set to config settings")
	}

	go func() {
		e.Logger.Fatal(e.Start(fmt.Sprintf(":%d", core.Deps.Config.Port)))
	}()

	// Set up graceful shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	core.Deps.Log.Info("Shutting down server...")
	close(shutdown)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := e.Shutdown(ctx); err != nil {
		core.Deps.Log.Fatalf("Server shutdown failed: %v", err)
	}
}

func CheckAlreadyRegistered(core *Core) bool {
	n, found := core.Neurons[core.Deps.Hotkey.Address]
	if !found {
		return false
	}
	var netip net.IP = n.AxonInfo.IP.Bytes()
	currentIp := fmt.Sprintf("http://%s:%d", netip, n.AxonInfo.Port)
	configIp := fmt.Sprintf("http://%s:%d", core.Deps.Config.Ip, core.Deps.Config.Port)
	return currentIp == configIp
}
