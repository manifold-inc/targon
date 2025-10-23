package vali

import (
	"fmt"
	"os"
	"strings"

	"targon/cli/shared"
	"targon/internal/cvm"

	"github.com/centrifuge/go-substrate-rpc-client/v4/signature"
	"github.com/manifold-inc/manifold-sdk/lib/utils"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var (
	logsIPFlag        string
	logsContainerFlag string
	logsTailFlag      string
)

func init() {
	valiCmd.AddCommand(logsCMD)
	logsCMD.Flags().StringVar(&logsIPFlag, "ip", "localhost", "IP address of the vm")
	logsCMD.Flags().StringVar(&logsContainerFlag, "container", "", "Name of the container to get logs from")
	logsCMD.Flags().StringVar(&logsTailFlag, "tail", "all", "Number of lines to show from the end of the logs")
}

var logsCMD = &cobra.Command{
	Use:   "logs",
	Short: "View logs for a vm",
	Long:  `View logs for a vm`,
	Run: func(cmd *cobra.Command, args []string) {
		if logsIPFlag == "" {
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
		if len(logsIPFlag) != 0 {
			cvmIP := strings.TrimPrefix(logsIPFlag, "http://")
			cvmIP = strings.TrimSuffix(cvmIP, ":8080")

			logs, err := attester.GetLogsFromNode(cvmIP, logsContainerFlag, logsTailFlag)
			if err != nil {
				fmt.Println(utils.Wrap("error getting logs from cvm", err))
				return
			}
			fmt.Printf("Logs:\n%s\n", logs)
			return
		}
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
