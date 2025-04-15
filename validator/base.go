package validator

import (
	"os"
	"os/signal"
	"syscall"

	"github.com/centrifuge/go-substrate-rpc-client/v4/types"
	"github.com/subtrahend-labs/gobt/client"
)

type BaseValidator struct {
	callbacks                   []func(types.Header)
	mainfunc                    func(i <-chan (bool), o chan<- (bool))
	onSubscriptionError         func(err error)
	onSubscriptionCreationError func(err error)
	NetUID                      int
}

func (b *BaseValidator) AddBlockCallback(f func(types.Header)) {
	b.callbacks = append(b.callbacks, f)
}

func (b *BaseValidator) SetMainFunc(f func(i <-chan (bool), o chan<- (bool))) {
	b.mainfunc = f
}

func (b *BaseValidator) SetOnSubscriptionError(f func(e error)) {
	b.onSubscriptionError = f
}

func (b *BaseValidator) SetOnSubscriptionCreationError(f func(e error)) {
	b.onSubscriptionCreationError = f
}

func NewValidator(n int) *BaseValidator {
	return &BaseValidator{NetUID: n}
}

func (b *BaseValidator) Start(c *client.Client) {
	// Handle graceful exits
	sigChan := make(chan os.Signal, 1)
	startCleanup := make(chan bool, 1)
	exitReady := make(chan bool, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	go b.mainfunc(startCleanup, exitReady)
	for {
		sub, err := c.Api.RPC.Chain.SubscribeNewHeads()
		if err != nil {
			b.onSubscriptionCreationError(err)
		}
		for {
			select {
			// Exit cleanly after system finishes current block
			case <-sigChan:
				// Send done signal
				startCleanup <- true
				// Wait for mainfunc to respond that it is done
				<-exitReady
				os.Exit(0)
			case head := <-sub.Chan():
				for _, exec := range b.callbacks {
					exec(head)
				}
			case err = <-sub.Err():
				b.onSubscriptionError(err)
				sub, err = c.Api.RPC.Chain.SubscribeNewHeads()
				if err != nil {
					b.onSubscriptionCreationError(err)
				}
			}
		}
	}
}
