package cli

import (
	"fmt"
	"math/big"
	"net/http"
	"os"
	"time"

	"targon/internal/setup"
	"targon/internal/targon"
	"targon/internal/utils"

	"github.com/centrifuge/go-substrate-rpc-client/v4/signature"
	"github.com/centrifuge/go-substrate-rpc-client/v4/types"
	"github.com/spf13/cobra"
	"github.com/subtrahend-labs/gobt/runtime"
	"go.uber.org/zap"
)

var (
	uidflag int
	ipflag  string
)

func init() {
	ipsCmd.Flags().IntVar(&uidflag, "uid", -1, "Specific uid to grab GPU info for")
	ipsCmd.Flags().StringVar(&ipflag, "ip", "", "Specific ip address for off chain testing")

	RootCmd.AddCommand(ipsCmd)
}

var ipsCmd = &cobra.Command{
	Use:   "attest",
	Short: "Manually attest a miner or ip address",
	Long:  ``,
	Run: func(cmd *cobra.Command, args []string) {
		deps := setup.Init(zap.FatalLevel)
		core := targon.CreateCore(deps)

		if uidflag == -1 && ipflag == "" {
			fmt.Println("Please specify uid or ip")
			return
		}
		if len(ipflag) != 0 {
			tr := &http.Transport{
				TLSHandshakeTimeout: 5 * time.Second,
				MaxConnsPerHost:     1,
				DisableKeepAlives:   true,
			}
			client := &http.Client{Transport: tr, Timeout: 5 * time.Minute}

			neuron := runtime.NeuronInfo{
				UID:    types.NewUCompact(big.NewInt(444)),
				Hotkey: types.AccountID(deps.Hotkey.PublicKey),
			}
			HOTKEY_PHRASE := os.Getenv("MINER_HOTKEY_PHRASE")
			kp, err := signature.KeyringPairFromSecret(HOTKEY_PHRASE, 42)
			if err == nil {
				fmt.Println("Using MINER_HOTKEY_PHRASE")
				neuron.Hotkey = types.AccountID(kp.PublicKey)
			} else {
				fmt.Println(utils.Wrap("Failed creating miner keyring par", err))
				fmt.Println("Using HOTKEY_PHRASE")
			}
			// Mock Neuron, use self hotkey
			gpus, _, _, err := targon.CheckCVMAttest(core, client, &neuron, ipflag)
			if err != nil {
				fmt.Println(utils.Wrap("CVM attest error", err))
				return
			}
			fmt.Printf("node: %s \n", ipflag)
			fmt.Printf("gpus: %v\n\n", gpus)
			return
		}

		blockHash, err := core.Deps.Client.Api.RPC.Chain.GetBlockHashLatest()
		if err != nil {
			fmt.Println(utils.Wrap("Failed getting blockhash for neurons", err))
			return
		}
		neuron, err := runtime.GetNeuron(core.Deps.Client, uint16(core.Deps.Env.NETUID), uint16(uidflag), &blockHash)
		if err != nil {
			fmt.Println(utils.Wrap("Failed getting neurons", err))
			return
		}

		tr := &http.Transport{
			TLSHandshakeTimeout: 5 * time.Second,
			MaxConnsPerHost:     1,
			DisableKeepAlives:   true,
		}
		client := &http.Client{Transport: tr, Timeout: 5 * time.Minute}

		nodes, err := targon.GetCVMNodes(core, client, neuron)
		if err != nil {
			fmt.Println(utils.Wrap("Failed to get nodes: %s", err))
			return
		}
		fmt.Printf("Nodes: %v\n", nodes)
		fmt.Println("CVM attest results")
		for _, n := range nodes {
			gpus, _, _, err := targon.CheckCVMAttest(core, client, neuron, n)
			if err != nil {
				fmt.Println(utils.Wrap("CVM attest error", err))
				continue
			}
			fmt.Printf("node: %s \n", n)
			fmt.Printf("gpus: %v\n\n", gpus)
		}
	},
}
