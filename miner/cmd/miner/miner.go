package main

import (
	"fmt"
	"net/http"
	"sync"

	"miner/cmd/internal/setup"

	"github.com/centrifuge/go-substrate-rpc-client/v4/types"
	"github.com/labstack/echo/v4"
	"github.com/subtrahend-labs/gobt/boilerplate"
)

type Core struct {
	Deps             *setup.Dependencies
	ValidatorPermits []types.Bool

	// Global core lock
	mu sync.Mutex
}

func CreateCore(d *setup.Dependencies) *Core {
	return &Core{
		Deps:             d,
		ValidatorPermits: []types.Bool{},
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
	validator.SetMainFunc(func(i <-chan bool, o chan<- bool) {
		e := echo.New()
		e.GET("/cvm", func(c echo.Context) error {
			// TODO verify sender is validator w/ vpermit
			sig := c.Request().Header.Get("Epistula-Request-Signature")
			timestamp := c.Request().Header.Get("Epistula-Timestamp")
			uuid := c.Request().Header.Get("Epistula-Uuid")
			signed_for := c.Request().Header.Get("Epistula-Signed-For")
			signed_by := c.Request().Header.Get("Epistula-Signed-By")
			boilerplate.VerifyEpistulaHeaders(
				core.Deps.Env.HOTKEY_SS58,
				sig,
				[]byte{},
				timestamp,
				uuid,
				signed_for,
				signed_by,
			)
			return c.JSON(http.StatusOK, core.Deps.Config.Nodes)
		})
		e.GET("/", func(c echo.Context) error {
			return c.String(http.StatusOK, "PONG")
		})
		e.Logger.Fatal(e.Start(":1323"))
	})

	validator.SetOnSubscriptionCreationError(func(e error) {
		deps.Log.Infow("Creation Error", "error", e)
		panic(e)
	})
	validator.SetOnSubscriptionError(func(e error) {
		deps.Log.Infow("Subscription Error", "error", e)
		panic(e)
	})
	validator.Start(deps.Client)
}
