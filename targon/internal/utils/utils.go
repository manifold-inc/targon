// Package utils
package utils

import (
	"errors"
	"fmt"

	"github.com/centrifuge/go-substrate-rpc-client/v4/types"
	"github.com/vedhavyas/go-subkey/v2"
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

func AccountIDToSS58(acc types.AccountID) string {
	recipientSS58 := subkey.SS58Encode(acc.ToBytes(), 42)
	return recipientSS58
}
