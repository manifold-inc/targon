package vali

import (
	"fmt"
	"os"
	"strings"

	"targon/internal/cvm"

	"github.com/centrifuge/go-substrate-rpc-client/v4/signature"
	"github.com/manifold-inc/manifold-sdk/lib/utils"
	"github.com/spf13/cobra"
)

var containersIPFlag string

func init() {
	valiCmd.AddCommand(containersCMD)
	containersCMD.Flags().StringVar(&containersIPFlag, "ip", "localhost", "IP address of the vm")
}

var containersCMD = &cobra.Command{
	Use:   "containers",
	Short: "Get containers from a vm",
	Long:  `Get containers from a vm`,
	Run: func(cmd *cobra.Command, args []string) {
		if containersIPFlag == "" {
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
		if len(containersIPFlag) != 0 {
			cvmIP := strings.TrimPrefix(containersIPFlag, "http://")
			cvmIP = strings.TrimSuffix(cvmIP, ":8080")

			containers, err := attester.GetContainers(cvmIP)
			if err != nil {
				fmt.Println(utils.Wrap("error getting containers from cvm", err))
				return
			}
			for _, con := range containers {
				name := strings.TrimPrefix(con.Names[0], "/")
				fmt.Println(name)
			}
			return
		}
	},
}
