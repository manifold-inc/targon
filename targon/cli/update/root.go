package update

import (
	"fmt"
	"targon/cli/root"

	"github.com/spf13/cobra"
)

func init() {
	root.RootCmd.AddCommand(updateCmd)
}

var updateCmd = &cobra.Command{
	Use:   "update",
	Short: "Update miner config",
	Long:  `Update miner config`,
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Println("Update miner config")
	},
}