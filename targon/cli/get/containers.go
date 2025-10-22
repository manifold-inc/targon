package get

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"targon/internal/cvm"

	"github.com/centrifuge/go-substrate-rpc-client/v4/signature"
	"github.com/manifold-inc/manifold-sdk/lib/utils"
	"github.com/spf13/cobra"
)

var containersIpFlag string

func init() {
	getCmd.AddCommand(containersCMD)
	containersCMD.Flags().StringVar(&containersIpFlag, "ip", "", "IP address of the vm")
}

var containersCMD = &cobra.Command{
	Use:   "containers",
	Short: "Get containers from a vm",
	Long:  `Get containers from a vm`,
	Run: func(cmd *cobra.Command, args []string) {
		if containersIpFlag == "" {
			_ = cmd.Help()
			return
		}

		config, err := loadConfig()
		if err != nil {
			fmt.Println("error loading config: " + err.Error())
			os.Exit(1)
		}
		
		kp, err := signature.KeyringPairFromSecret(config.ValidatorHotkeyPhrase, 42)
		if err != nil {
			fmt.Println("error parsing hotkey phrase: " + err.Error())
			os.Exit(1)
		}

		attester := cvm.NewAttester(1, kp, "https://tower.targon.com")
		if len(containersIpFlag) != 0 {
			cvmIP := strings.TrimPrefix(containersIpFlag, "http://")
			cvmIP = strings.TrimSuffix(cvmIP, ":8080")

			containers, err := attester.GetContainers(cvmIP)
			if err != nil {
				fmt.Println(utils.Wrap("error getting containers from cvm", err))
				return
			}
			json, err := json.MarshalIndent(containers, "", "  ")
			if err != nil {
				fmt.Println(utils.Wrap("error marshalling containers response", err))
				return
			}
			fmt.Printf("Containers:\n%s\n", string(json))
			return
		}
	},
}
