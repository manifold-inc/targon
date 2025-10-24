// Package vali
package vali

import (
	"os"

	"targon/cli/root"

	"github.com/spf13/cobra"
)

func init() {
	root.RootCmd.AddCommand(valiCmd)
}

var valiCmd = &cobra.Command{
	Use:     "vali",
	Short:   "Validator related commands",
	Aliases: []string{"v"},
	Long:    `Validator related commands`,
	Run: func(cmd *cobra.Command, args []string) {
		if len(args) == 0 {
			_ = cmd.Help()
			os.Exit(0)
		}
	},
}
