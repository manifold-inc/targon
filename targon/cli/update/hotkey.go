package update

import (
	"fmt"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var hotkeyflag string

func init() {
	hotkeyCMD.Flags().StringVar(&hotkeyflag, "hotkey", "", "Miner hotkey to update to")
	updateCmd.AddCommand(hotkeyCMD)
}

var hotkeyCMD = &cobra.Command{
	Use:   "hotkey",
	Short: "Update the hotkey for a miner",
	Long:  `Update the hotkey for a miner`,
	Run: func(cmd *cobra.Command, args []string) {
		if hotkeyflag == "" {
			fmt.Println("No hotkey provided")
			return
		}

		viper.Set("miner_hotkey", hotkeyflag)
		err := viper.WriteConfig()
		if err != nil {
			fmt.Printf("Failed to write config: %v\n", err)
			fmt.Printf("Config file path: %s\n", viper.ConfigFileUsed())
			return
		}

		fmt.Println("Hotkey updated")
	},
}