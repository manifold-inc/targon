package callbacks

import (
	"encoding/binary"
	"errors"
	"math/big"
	"net"
	"time"

	"targon/internal/setup"

	"github.com/centrifuge/go-substrate-rpc-client/v4/types"
	"github.com/subtrahend-labs/gobt/extrinsics"
	"github.com/subtrahend-labs/gobt/sigtools"
)

func ServeToChain(d *setup.Dependencies) error {
	netuid := types.NewU16(uint16(d.Env.NETUID))

	// Not sure what this is used for, just setting this to targon ver
	version := types.NewU32(uint32(d.Env.VERSION))

	netip := net.ParseIP(d.Env.VALI_IP)
	if netip == nil {
		return errors.New("could not parse ip")
	}

	ip := types.NewU128(*big.NewInt(int64(binary.BigEndian.Uint32(netip.To4()))))

	port := types.NewU16(uint16(80))
	ipType := types.NewU8(4)   // IPv4
	protocol := types.NewU8(4) // HTTP
	placeholder1 := types.NewU8(0)
	placeholder2 := types.NewU8(0)

	// Create and submit the ServeAxon extrinsic
	ext, err := extrinsics.ServeAxonExt(
		d.Client,
		netuid,
		version,
		ip,
		port,
		ipType,
		protocol,
		placeholder1,
		placeholder2,
	)
	if err != nil {
		return err
	}
	ops, err := sigtools.CreateSigningOptions(d.Client, d.Hotkey, nil)
	if err != nil {
		return err
	}
	err = ext.Sign(
		d.Hotkey,
		d.Client.Meta,
		ops...,
	)
	if err != nil {
		return err
	}

	sub, err := d.Client.Api.RPC.Author.SubmitAndWatchExtrinsic(*ext)
	if err != nil {
		return err
	}
	defer sub.Unsubscribe()

	tout := make(chan bool, 1)
	go func() {
		time.Sleep(60 * time.Second)
		tout <- true
	}()
	d.Log.Info("Waiting for posted serve info")
	for {
		select {
		case status := <-sub.Chan():
			if status.IsInBlock {
				d.Log.Info("Posted vali info to chain successfully")
				return nil
			}
			if status.IsDropped || status.IsInvalid || status.IsRetracted || status.IsUsurped {
				return errors.New("failed setting ip info")
			}
		case err := <-sub.Err():
			if err != nil {
				d.Log.Error(err.Error())
			}
			return err
		case <-tout:
			return errors.New("setting ip info timedout")
		}
	}
}
