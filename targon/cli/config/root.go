package config

import (
	"fmt"

	"targon/cli/root"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var netuidflag int
var hotkeyflag string
var nvidia_attest_endpoint_flag string

func init() {
	configCmd.Flags().StringVar(&hotkeyflag, "hotkey", "", "Hotkey phrase to update to")
	configCmd.Flags().IntVar(&netuidflag, "netuid", 0, "Netuid to update to")
	configCmd.Flags().StringVar(&nvidia_attest_endpoint_flag, "nvidia_attest_endpoint", "", "NVIDIA attest endpoint to update to")
	root.RootCmd.AddCommand(configCmd)
}

var configCmd = &cobra.Command{
	Use:   "config",
	Short: "Update config values",
	Long:  `Update one or more configuration values. Use flags to specify which values to update.`,
	Run: func(cmd *cobra.Command, args []string) {
		updated := false

		if hotkeyflag != "" {
			viper.Set("HOTKEY_PHRASE", hotkeyflag)
			fmt.Printf("Hotkey phrase updated to: %s\n", hotkeyflag)
			updated = true
		}

		if netuidflag != 0 {
			viper.Set("netuid", netuidflag)
			fmt.Printf("Netuid updated to: %d\n", netuidflag)
			updated = true
		}

		if nvidia_attest_endpoint_flag != "" {
			viper.Set("nvidia_attest_endpoint", nvidia_attest_endpoint_flag)
			fmt.Printf("NVIDIA attest endpoint updated to: %s\n", nvidia_attest_endpoint_flag)
			updated = true
		}

		if !updated {
			fmt.Println("No configuration values specified to update.")
			fmt.Println("Use --help to see available options.")
			return
		}

		err := viper.WriteConfig()
		if err != nil {
			fmt.Printf("Failed to write config: %v\n", err)
			fmt.Printf("Config file path: %s\n", viper.ConfigFileUsed())
			return
		}
	},
}

