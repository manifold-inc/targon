package get

import (
	"fmt"
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
		fmt.Println("Get targon info")
	},
}
