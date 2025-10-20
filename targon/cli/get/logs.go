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
	ipFlag            string
	containerNameFlag string
	tailFlag          string
)

func init() {
	getCmd.AddCommand(logsCMD)
	logsCMD.Flags().StringVar(&ipFlag, "ip", "", "Specific ip address for off chain testing")
	logsCMD.Flags().StringVar(&containerNameFlag, "container-name", "", "Name of the container to get logs from")
	logsCMD.Flags().StringVar(&tailFlag, "tail", "all", "Number of lines to show from the end of the logs")
}

var logsCMD = &cobra.Command{
	Use:   "logs",
	Short: "View logs for a vm",
	Long:  `View logs for a vm`,
	Run: func(cmd *cobra.Command, args []string) {
		if ipFlag == "" {
			_ = cmd.Help()
			return
		}

		kp, err := signature.KeyringPairFromSecret(viper.GetString("validator.hotkey_phrase"), 42)
		if err != nil {
			fmt.Println("error parsing hotkey phrase: " + err.Error())
			os.Exit(1)
		}

		attester := cvm.NewAttester(1, kp, "https://tower.targon.com")
		if len(ipFlag) != 0 {
			cvmIP := strings.TrimPrefix(ipFlag, "http://")
			cvmIP = strings.TrimSuffix(cvmIP, ":8080")

			logs, err := attester.GetLogsFromNode(cvmIP, containerNameFlag, tailFlag)
			if err != nil {
				fmt.Println(utils.Wrap("error getting logs from cvm", err))
				return
			}
			fmt.Printf("Logs:\n%s\n", logs)
			return
		}
	},
}
