package config

import (
	"fmt"

	"targon/cli/root"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var netuidFlag int
var minerHotkeyPhraseFlag string
var validatorHotkeyPhraseFlag string
var nvidiaAttestEndpointFlag string

func init() {
	configCmd.Flags().StringVar(&minerHotkeyPhraseFlag, "miner_hotkey_phrase", "", "Miner hotkey phrase to update to")
	configCmd.Flags().StringVar(&validatorHotkeyPhraseFlag, "validator_hotkey_phrase", "", "Validator hotkey phrase to update to")
	configCmd.Flags().IntVar(&netuidFlag, "netuid", -1, "Netuid to update to")
	configCmd.Flags().StringVar(&nvidiaAttestEndpointFlag, "nvidia_attest_endpoint", "", "NVIDIA attest endpoint to update to")
	root.RootCmd.AddCommand(configCmd)
}

var configCmd = &cobra.Command{
	Use:   "config",
	Short: "Update config values",
	Long:  `Update one or more configuration values. Use flags to specify which values to update.`,
	Run: func(cmd *cobra.Command, args []string) {
		updated := false

		if minerHotkeyPhraseFlag != "" {
			viper.Set("miner.hotkey_phrase", minerHotkeyPhraseFlag)
			fmt.Printf("Miner hotkey phrase updated to: %s\n", minerHotkeyPhraseFlag)
			updated = true
		}

		if validatorHotkeyPhraseFlag != "" {
			viper.Set("validator.hotkey_phrase", validatorHotkeyPhraseFlag)
			fmt.Printf("Validator hotkey phrase updated to: %s\n", validatorHotkeyPhraseFlag)
			updated = true
		}

		if netuidFlag != 0 {
			viper.Set("netuid", netuidFlag)
			fmt.Printf("Netuid updated to: %d\n", netuidFlag)
			updated = true
		}

		if nvidiaAttestEndpointFlag != "" {
			viper.Set("nvidia_attest_endpoint", nvidiaAttestEndpointFlag)
			fmt.Printf("NVIDIA attest endpoint updated to: %s\n", nvidiaAttestEndpointFlag)
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
