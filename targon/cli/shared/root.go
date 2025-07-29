package shared

import (
	"bufio"
	"fmt"
	"os"
	"strconv"

	"github.com/spf13/viper"
)

func PromptConfigString(key string) string {
	fmt.Printf("Enter your %s: ", key)
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Scan()

	viper.Set(key, scanner.Text())
	if err := viper.WriteConfig(); err != nil {
		fmt.Println("Error writing config: " + err.Error())
		os.Exit(1)
	}
	return viper.GetString(key)
}

func PromptConfigInt(key string) int {
	fmt.Printf("Enter your %s: ", key)
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Scan()

	num, err := strconv.Atoi(scanner.Text())
	if err != nil {
		fmt.Println("Error parsing integer: " + err.Error())
		os.Exit(1)
	}

	viper.Set(key, num)
	if err := viper.WriteConfig(); err != nil {
		fmt.Println("Error writing config: " + err.Error())
		os.Exit(1)
	}
	return num
}
