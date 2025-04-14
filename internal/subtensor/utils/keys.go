package utils

import (
	"github.com/centrifuge/go-substrate-rpc-client/v4/types"
	"github.com/vedhavyas/go-subkey/v2"
)

func AccountIDToSS58(acc types.AccountID) string {
	recipientSS58 := subkey.SS58Encode(acc.ToBytes(), 42)
	return recipientSS58
}
