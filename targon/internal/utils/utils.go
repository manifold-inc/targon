package utils

import (
	"errors"
	"fmt"
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
