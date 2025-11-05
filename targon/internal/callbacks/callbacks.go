// Package callbacks
package callbacks

import (
	"errors"
	"fmt"
	"math/rand"
	"net"
	"time"

	"targon/internal/targon"

	"github.com/centrifuge/go-substrate-rpc-client/v4/types"
	"github.com/subtrahend-labs/gobt/boilerplate"
	"github.com/subtrahend-labs/gobt/storage"
)

func CheckAlreadyRegistered(core *targon.Core) error {
	uid, found := core.HotkeyToUID[core.Deps.Hotkey.Address]
	if !found {
		return errors.New("not registered on sn")
	}
	n, found := core.Neurons[uid]
	if !found {
		return errors.New("not registered on sn")
	}
	var netip net.IP = n.AxonInfo.IP.Bytes()
	currentIP := netip.String()
	configIP := core.Deps.Env.ValiIP
	if configIP == currentIP {
		return nil
	}
	return fmt.Errorf("ip %s does not match chain ip %s", configIP, currentIP)
}

func AddBlockCallbacks(v *boilerplate.BaseChainSubscriber, c *targon.Core) {
	// Wrapped for closure
	getBlocksFrom := func(b types.Header) int {
		tempo := 360
		return ((tempo + 1 + c.Deps.Env.Netuid + 1 + int(b.Number)) % (tempo + 1))
	}

	// block timer for catching hangs
	t := time.AfterFunc(1*time.Hour, func() {
		c.Deps.Log.Error("havint seen any blocks in over an hour, am i stuck?")
	})
	v.AddBlockCallback(func(h types.Header) {
		t.Reset(1 * time.Hour)
	})

	// Logging blocks
	v.AddBlockCallback(func(h types.Header) {
		if c.StartupBlock == 0 {
			c.StartupBlock = int(h.Number)
		}
		c.Deps.Log.Infow(
			"New block",
			"block",
			fmt.Sprintf("%v", h.Number),
			"left_in_interval",
			361-getBlocksFrom(h),
		)
	})

	// get neurons and set weights if needed
	// dont check reg if debug is true
	hasCheckedReg := c.Deps.Env.Debug
	v.AddBlockCallback(func(h types.Header) {
		// Run after second block of interval
		if getBlocksFrom(h) != 2 && len(c.Neurons) != 0 {
			return
		}
		getNeuronsCallback(c, h)
		if !hasCheckedReg {
			if err := CheckAlreadyRegistered(c); err != nil {
				c.Deps.Log.Infof("Setting validator info, differs from config: %w", err)
				err := ServeToChain(c.Deps)
				if err != nil {
					c.Deps.Log.Errorw("Failed serving extrinsic", "error", err)
				}
			} else {
				c.Deps.Log.Info("Skipping set miner info, already set to config settings")
			}
		}
		hasCheckedReg = true
	})

	// get emission and auction data for this interval
	v.AddBlockCallback(func(h types.Header) {
		if c.EmissionPool != nil {
			return
		}
		// Get tower pyth price and emission slice along with min burn
		auctionData, err := c.Deps.Tower.AuctionDetails()
		if err != nil {
			c.Deps.Log.Errorw("Failed getting tao price", "error", err)
			return
		}
		c.TaoPrice = &auctionData.TaoPrice
		c.Auctions = auctionData.Auctions
		c.BurnDistribution = auctionData.BurnDistribution
		c.Deps.Log.Infof("Auctions: %+v", c.Auctions)
		c.Deps.Log.Infof("Current tao price $%f", *c.TaoPrice)
		alphaOut, err := storage.GetSubnetAlphaOutEmission(c.Deps.Client, types.NewU16(uint16(c.Deps.Env.Netuid)), &h.ParentHash)
		if err != nil {
			c.Deps.Log.Errorw("Validator is falling behind current block time")
			alphaOut, err = storage.GetSubnetAlphaOutEmission(c.Deps.Client, types.NewU16(uint16(c.Deps.Env.Netuid)), nil)
			if err != nil {
				c.Deps.Log.Errorw("Failed getting sn tao emissions", "error", err)
				return
			}
		}
		price, err := storage.GetSubnetMovingPrice(c.Deps.Client, types.NewU16(uint16(c.Deps.Env.Netuid)), &h.ParentHash)
		if err != nil {
			c.Deps.Log.Errorw("Validator is falling behind current block time")
			price, err = storage.GetSubnetMovingPrice(c.Deps.Client, types.NewU16(uint16(c.Deps.Env.Netuid)), nil)
			if err != nil {
				c.Deps.Log.Errorw("Failed getting sn tao emissions", "error", err)
				return
			}
		}
		emi := (float64(*alphaOut) / 1e9) * .41 * 360 * *c.TaoPrice * price.Float64()
		// Protect against zero emi
		if emi < 1 {
			emi = 1
		}
		c.EmissionPool = &emi
		c.Deps.Log.Infof("Current sn miner emission pool in $ %f", *c.EmissionPool)
	})

	// get miner nodes
	// Every 30 blocks off the internval tempo untill 59 left in block
	v.AddBlockCallback(func(h types.Header) {
		if (getBlocksFrom(h)%30 != 1 || getBlocksFrom(h) > 301) && len(c.MinerNodes) != 0 {
			return
		}
		getNodesAll(c)
	})

	// get passing attestations
	v.AddBlockCallback(func(h types.Header) {
		if c.Neurons == nil {
			return
		}
		if 361-getBlocksFrom(h) < 20 {
			return
		}
		// Not on specific tempo;
		// helps reduce stress on cvm nodes from number of pings
		chance := rand.Float64()
		if chance < c.Deps.Env.AttestRate && len(c.VerifiedNodes) != 0 {
			return
		}
		getPassingAttestations(c)
	})

	// Log weights
	v.AddBlockCallback(func(h types.Header) {
		if h.Number%10 != 0 || len(c.MinerNodes) == 0 {
			return
		}
		uids, scores, _, _ := getWeights(c)
		c.Deps.Log.Infow(
			"Current Weights",
			"uids",
			fmt.Sprintf("%+v", uids),
			"scores",
			fmt.Sprintf("%+v", scores),
		)
	})

	// Set Weights
	v.AddBlockCallback(func(h types.Header) {
		if getBlocksFrom(h) != 1 {
			return
		}
		if len(c.MinerNodes) == 0 {
			c.Deps.Log.Warn("Skipping weightset from no miner nodes")
			return
		}
		if int(h.Number)-c.StartupBlock < 180 {
			c.Deps.Log.Warn("Skipping weightset from recent startup block")
			return
		}
		uids, scores, results, err := getWeights(c)
		if err != nil {
			c.Deps.Log.Errorw("Failed getting weights", "error", err)
			return
		}
		c.AuctionResults = results

		err = sendIntervalSummary(c, h, uids, scores)
		if err != nil {
			c.Deps.Log.Warnw("Failed logging to discord", "error", err)
		}
		if c.Deps.Mongo != nil {
			syncErr := targon.SyncMongo(c, uids, scores, h)
			if syncErr != nil {
				c.Deps.Log.Errorw("Failed syncing complete data to mongo", "error", syncErr)
			}

			if syncErr == nil {
				c.Deps.Log.Infow("Weights set and complete data synced",
					"uids", uids,
					"scores", scores,
					"block", h.Number,
					"miners", len(uids))
			}
		}
		setWeights(c, uids, scores)

		// catches up to live block time
		block, err := c.Deps.Client.Api.RPC.Chain.GetBlockLatest()
		if err != nil {
			c.Deps.Log.Infow("Failed getting latest block, restarting")
			v.Restart()
		}
		if block.Block.Header.Number-h.Number > 15 {
			c.Deps.Log.Warnw("Blocks drifting, resetting to live blocktime")
			v.Restart()
		}
	})
}
