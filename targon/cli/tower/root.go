package tower

import (
	"fmt"
	"net/http"
	"time"

	"targon/cli/root"
	"targon/internal/setup"
	"targon/internal/targon"
	tw "targon/internal/tower"

	"github.com/spf13/cobra"
	"go.uber.org/zap"
)

var ipflag string

func init() {
	towerCmd.Flags().StringVar(&ipflag, "ip", "", "Specific ip address for off chain testing")

	root.RootCmd.AddCommand(towerCmd)
}

var towerCmd = &cobra.Command{
	Use:   "tower",
	Short: "Manually attest a miner or ip address",
	Long:  ``,
	Run: func(cmd *cobra.Command, args []string) {
		if ipflag == "" {
			cmd.Help()
			return
		}
		deps := setup.Init(zap.FatalLevel)
		core := targon.CreateCore(deps)
		towerClient := &http.Client{Transport: &http.Transport{
			TLSHandshakeTimeout: 5 * time.Second,
			DisableKeepAlives:   true,
		}, Timeout: 1 * time.Minute}
		tower := tw.NewTower(towerClient, "https://tower.targon.com", &core.Deps.Hotkey, core.Deps.Log)
		res := tower.Check(ipflag)
		fmt.Printf("%t\n", res)
	},
}
