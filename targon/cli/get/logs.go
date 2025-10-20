package get

import (
	"fmt"
	"os"
	"strings"

	"targon/internal/cvm"

	"github.com/centrifuge/go-substrate-rpc-client/v4/signature"
	"github.com/manifold-inc/manifold-sdk/lib/utils"
	"github.com/spf13/viper"

	"github.com/spf13/cobra"
)

var (
	logsHotkeyFlag    string
	logsIpFlag        string
	containerNameFlag string
	tailFlag          string
)

func init() {
	getCmd.AddCommand(logsCMD)
	logsCMD.Flags().StringVar(&logsIpFlag, "ip", "", "IP address of the vm")
	logsCMD.Flags().StringVar(&containerNameFlag, "container", "", "Name of the container to get logs from")
	logsCMD.Flags().StringVar(&tailFlag, "tail", "all", "Number of lines to show from the end of the logs")
	logsCMD.Flags().StringVar(&logsHotkeyFlag, "hotkey", "", "Miner hotkey")
}

var logsCMD = &cobra.Command{
	Use:   "logs",
	Short: "View logs for a vm",
	Long:  `View logs for a vm`,
	Run: func(cmd *cobra.Command, args []string) {
		if logsIpFlag == "" || logsHotkeyFlag == "" {
			_ = cmd.Help()
			return
		}

		kp, err := signature.KeyringPairFromSecret(viper.GetString("validator.hotkey_phrase"), 42)
		if err != nil {
			fmt.Println("error parsing hotkey phrase: " + err.Error())
			os.Exit(1)
		}

		attester := cvm.NewAttester(1, kp, "https://tower.targon.com")
		if len(logsIpFlag) != 0 {
			cvmIP := strings.TrimPrefix(logsIpFlag, "http://")
			cvmIP = strings.TrimSuffix(cvmIP, ":8080")

			logs, err := attester.GetLogsFromNode(logsHotkeyFlag, cvmIP, containerNameFlag, tailFlag)
			if err != nil {
				fmt.Println(utils.Wrap("error getting logs from cvm", err))
				return
			}
			fmt.Printf("Logs:\n%s\n", logs)
			return
		}
	},
}
