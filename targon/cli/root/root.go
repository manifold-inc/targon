package root

import (
	"bufio"
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
			fmt.Println("Config file not found, creating new one at ~/.config/.targon.json")
			initConfig()
		} else {
			panic(fmt.Errorf("Fatal error config file: %w", err))
		}
	}
}

func initConfig() error {
	home, err := os.UserHomeDir()
	if err != nil {
		return fmt.Errorf("Failed to get home directory: %w", err)
	}
	
	config := filepath.Join(home, ".config")
	path := filepath.Join(config, ".targon.json")
	
	// 0755: Default permissions for directory rwx-rx-rx for owner-group-others
	if err := os.MkdirAll(config, 0755); err != nil {
		return fmt.Errorf("Failed to create config directory: %w", err)
	}
	
	fmt.Print("Enter your HOTKEY PHRASE: ")
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Scan()
	HOTKEY_PHRASE := scanner.Text()
	
	viper.Set("HOTKEY_PHRASE", HOTKEY_PHRASE)
	if err := viper.WriteConfigAs(path); err != nil {
		return fmt.Errorf("Failed to write config file: %w", err)
	}
	
	fmt.Printf("Config file created successfully at %s\n", path)
	return nil
}

var RootCmd = &cobra.Command{
	Use:   "targon",
	Short: "",
	Long:  ``,
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Println("TODO docs")
	},
}

func Execute() {
	if err := RootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}