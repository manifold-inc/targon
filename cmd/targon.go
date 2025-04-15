package main

import (
	"fmt"
	"time"

	"targon/internal/setup"
	"targon/validator"

	"github.com/centrifuge/go-substrate-rpc-client/v4/types"
)

func main() {
	deps := setup.Init()
	deps.Log.Infof("Starting validator with key [%s] on chain [%s]", deps.Env.HOTKEY_SS58, deps.Env.CHAIN_ENDPOINT)

	validator := validator.NewValidator(4)
	deps.Log.Infof("Creating validator on netuid [%d]", validator.NetUID)
	validator.AddBlockCallback(func(h types.Header) {
		deps.Log.Infow("New block", "block", fmt.Sprintf("%v", h.Number))
	})
	validator.SetOnSubscriptionCreationError(func(e error) {
		deps.Log.Infow("Creation Error", "error", e)
		panic(e)
	})
	validator.SetOnSubscriptionError(func(e error) {
		deps.Log.Infow("Subscription Error", "error", e)
		panic(e)
	})
	validator.SetMainFunc(func(i <-chan bool, o chan<- bool) {
		for {
			select {
			case <-i:
				deps.Log.Info("See shutdown call")
				time.Sleep(5 * time.Second)
				o <- true
				return
			default:
				deps.Log.Info("Running....")
				time.Sleep(2 * time.Second)
			}
		}
	})
	validator.Start(deps.Client)

	//neurons, err := runtime.GetNeurons(deps.Client, uint16(netuid), &blockHash)
	//if err != nil {
	//	deps.Log.Infof("Error fetching neurons for netuid [%d]: [%s]", netuid, err)
	//}

	//deps.Log.Infof("total of %d neurons", len(neurons))
	//for i, neuron := range neurons {
	//	deps.Log.Infow(
	//		"neuron",
	//		"Neuron",
	//		strconv.Itoa(i),
	//		"Hotkey",
	//		utils.AccountIDToSS58(neuron.Hotkey),
	//		"Coldkey",
	//		utils.AccountIDToSS58(neuron.Coldkey),
	//		"UID",
	//		fmt.Sprintf("%d", neuron.UID.Int64()),
	//		"Active",
	//		fmt.Sprintf("%v", neuron.Active == types.NewBool(true)),
	//	)
	//}
}
