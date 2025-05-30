package get

import (
	"fmt"
	"net/http"
	"targon/internal/setup"
	"targon/internal/targon"
	"targon/internal/utils"
	"time"

	"github.com/spf13/cobra"
	"github.com/subtrahend-labs/gobt/runtime"
	"go.uber.org/zap"
)

var uid int

func init() {
	ipsCmd.Flags().IntVar(&uid, "uid", -1, "Specific uid to grab GPU info for")
	getCmd.AddCommand(ipsCmd)
}

var ipsCmd = &cobra.Command{
	Use:   "ips",
	Short: "",
	Long:  ``,
	Run: func(cmd *cobra.Command, args []string) {
		deps := setup.Init(zap.FatalLevel)
		core := targon.CreateCore(deps)

		blockHash, err := core.Deps.Client.Api.RPC.Chain.GetBlockHashLatest()
		if err != nil {
			fmt.Println(utils.Wrap("Failed getting blockhash for neurons", err))
			return
		}
		var neurons []runtime.NeuronInfo
		switch uid {
		case -1:
			neurons, err = runtime.GetNeurons(core.Deps.Client, uint16(core.Deps.Env.NETUID), &blockHash)
			if err != nil {
				fmt.Println(utils.Wrap("Failed getting neurons", err))
				return
			}
		default:
			neuron, err := runtime.GetNeuron(core.Deps.Client, uint16(core.Deps.Env.NETUID), uint16(uid), &blockHash)
			if err != nil {
				fmt.Println(utils.Wrap("Failed getting neuron", err))
				return
			}
			neurons = append(neurons, *neuron)
		}
		tr := &http.Transport{
			TLSHandshakeTimeout: 5 * time.Second,
			MaxConnsPerHost:     1,
			DisableKeepAlives:   true,
		}
		client := &http.Client{Transport: tr, Timeout: 5 * time.Minute}
		for _, neuron := range neurons {
			nodes, err := targon.GetCVMNodes(core, client, &neuron)
			if err != nil {
				continue
			}
			fmt.Printf("UID %d GPU Info:\n", neuron.UID.Int64())
			for _, node := range nodes {
				fmt.Printf("%s\n", node)
			}
			fmt.Println()
		}
	},
}
