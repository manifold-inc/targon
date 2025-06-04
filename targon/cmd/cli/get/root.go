package get

import (
	"fmt"
	"targon/cmd/cli"

	"github.com/spf13/cobra"
)

func init() {
	cli.RootCmd.AddCommand(getCmd)
}

var getCmd = &cobra.Command{
	Use:   "get",
	Short: "g",
	Long:  ``,
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Println("Get targon info")
	},
}
