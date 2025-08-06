package root

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

func init() {
	viper.SetConfigName(".targon")
	viper.SetConfigType("json")
	viper.AddConfigPath("$HOME/.config")

	if err := viper.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); ok {
			fmt.Println("config file not found, creating new one at ~/.config/.targon.json")
			err := initConfig()
			if err != nil {
				panic(err)
			}
		} else {
			panic(fmt.Errorf("fatal error config file: %w", err))
		}
	}
}

func initConfig() error {
	home, err := os.UserHomeDir()
	if err != nil {
		return fmt.Errorf("failed to get home directory: %w", err)
	}

	config := filepath.Join(home, ".config")
	path := filepath.Join(config, ".targon.json")

	// 0755: Default permissions for directory rwx-rx-rx for owner-group-others
	if err := os.MkdirAll(config, 0755); err != nil {
		return fmt.Errorf("failed to create config directory: %w", err)
	}

	if err := viper.WriteConfigAs(path); err != nil {
		return fmt.Errorf("failed to write config file: %w", err)
	}

	fmt.Printf("Config file created successfully at %s\n", path)
	return nil
}

var RootCmd = &cobra.Command{
	Use:   "targon",
	Short: "",
	Long:  ``,
	Run: func(cmd *cobra.Command, args []string) {
		if len(args) == 0 {
			cmd.Help()
			os.Exit(0)
		}
	},
}

func Execute() {
	if err := RootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}
