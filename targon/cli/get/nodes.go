package get

import (
	"fmt"
	"net"
	"os"

	"targon/cli/shared"
	"targon/internal/cvm"
	"targon/internal/utils"

	"github.com/centrifuge/go-substrate-rpc-client/v4/signature"
	"github.com/centrifuge/go-substrate-rpc-client/v4/types"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"github.com/subtrahend-labs/gobt/client"
	"github.com/subtrahend-labs/gobt/runtime"
)

var (
	getUIDflag        int
	getIPFlag         string
	getHotkeyFlag     string
	chainEndpointFlag string
)

func init() {
	getCMD.Flags().IntVar(&getUIDflag, "uid", -1, "Specific uid to grab GPU info for")
	getCMD.Flags().StringVar(&getHotkeyFlag, "hotkey", "", "hotkey of miner")
	getCMD.Flags().StringVar(&getIPFlag, "ip", "", "ip of miner")
	getCMD.Flags().StringVar(&chainEndpointFlag, "chain", "wss://entrypoint-finney.opentensor.ai:443", "Set chain endpoint")
	getCmd.AddCommand(getCMD)
}

var getCMD = &cobra.Command{
	Use:   "nodes",
	Short: "Get nodes from miner",
	Long:  `Get nodes from miner`,
	Run: func(cmd *cobra.Command, args []string) {
		if getUIDflag == -1 && getIPFlag == "" {
			_ = cmd.Help()
			return
		}
		config, err := loadConfig()
		if err != nil {
			fmt.Println("Error loading config: " + err.Error())
			os.Exit(1)
		}
		client, err := client.NewClient(chainEndpointFlag)
		if err != nil {
			fmt.Printf("Error creating client: %s\n", err)
			os.Exit(1)
		}
		kp, err := signature.KeyringPairFromSecret(config.ValidatorHotkeyPhrase, 42)
		if err != nil {
			fmt.Println("Error parsing hotkey phrase: " + err.Error())
			os.Exit(1)
		}
		var hotkey string
		var ip string
		if getUIDflag != -1 {
			blockHash, err := client.Api.RPC.Chain.GetBlockHashLatest()
			if err != nil {
				fmt.Println(utils.Wrap("Failed getting blockhash for neurons", err))
				return
			}
			neuron, err := runtime.GetNeuron(client, uint16(4), uint16(getUIDflag), &blockHash)
			if err != nil {
				fmt.Println(utils.Wrap("Failed getting neurons", err))
				return
			}
			hotkey = utils.AccountIDToSS58(neuron.Hotkey)
			var neuronIPAddr net.IP = neuron.AxonInfo.IP.Bytes()
			ip = fmt.Sprintf("%s:%d", neuronIPAddr.String(), neuron.AxonInfo.Port)
		}
		if getUIDflag == -1 {
			hotkey = utils.AccountIDToSS58(types.AccountID(kp.PublicKey))
			if getHotkeyFlag != "" {
				hotkey = getHotkeyFlag
			}
			ip = getIPFlag
		}
		attester := cvm.NewAttester(1, kp, "https://tower.targon.com")
		nodes, err := attester.GetNodes(hotkey, ip)
		if err != nil {
			fmt.Println(err.Error())
			return
		}
		fmt.Printf("%+v", nodes)
	},
}

type GetConfig struct {
	ValidatorHotkeyPhrase string
}

func loadConfig() (*GetConfig, error) {
	config := &GetConfig{}

	configStrings := map[string]*string{
		"validator.hotkey_phrase": &config.ValidatorHotkeyPhrase,
	}

	for key, value := range configStrings {
		if viper.GetString(key) == "" {
			shared.PromptConfigString(key)
		}
		*value = viper.GetString(key)
	}

	return config, nil
}
