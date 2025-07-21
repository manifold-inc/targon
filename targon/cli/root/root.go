package root

import (
	"fmt"
	"os"

	"github.com/joho/godotenv"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

func init() {
	viper.SetConfigName(".targon")
	viper.SetConfigType("json")
	viper.AddConfigPath("$HOME/.config")

	if err := viper.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); ok {
			fmt.Println("No config file found, please init config file at ~/.config/.targon.json")
			os.Exit(1)
		} else {
			panic(fmt.Errorf("fatal error config file: %w", err))
		}
	}

	_ = godotenv.Load()
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