package get

import (
	"os"
	"targon/cli/root"

	"github.com/spf13/cobra"
)

func init() {
	root.RootCmd.AddCommand(getCmd)
}

var getCmd = &cobra.Command{
	Use:   "get",
	Short: "Fetch data from mongo / chain",
	Long:  `Fetch data from mongo or chain and display it in various formats`,
	Run: func(cmd *cobra.Command, args []string) {
		if len(args) == 0 {
			cmd.Help()
			os.Exit(0)
		}
	},
}
