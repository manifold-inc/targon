package update

import (
	"fmt"

	"targon/cli/root"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var hotkeyflag string

func init() {
	updateCmd.Flags().StringVar(&hotkeyflag, "hotkey", "", "Hotkey phrase to update to")
	root.RootCmd.AddCommand(updateCmd)
}

var updateCmd = &cobra.Command{
	Use:   "config",
	Short: "Update config",
	Long:  `Update config`,
	Run: func(cmd *cobra.Command, args []string) {
		if hotkeyflag == "" {
			fmt.Println("No hotkey phrase provided")
			return
		}

		viper.Set("HOTKEY_PHRASE", hotkeyflag)
		err := viper.WriteConfig()
		if err != nil {
			fmt.Printf("Failed to write config: %v\n", err)
			fmt.Printf("Config file path: %s\n", viper.ConfigFileUsed())
			return
		}
		fmt.Printf("Hotkey phrase successfully updated to %s\n", hotkeyflag)
	},
}

