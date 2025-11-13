package vali

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"targon/cli/shared"

	"github.com/centrifuge/go-substrate-rpc-client/v4/signature"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"github.com/subtrahend-labs/gobt/boilerplate"
)

var (
	updateIPFlag  string
	updateEnvFlag string
)

func init() {
	valiCmd.AddCommand(updateCmd)
	updateCmd.Flags().StringVar(&updateIPFlag, "ip", "localhost", "IP address of the vm")
	updateCmd.Flags().StringVar(&updateEnvFlag, "env", "", "path to .env file")
}

var updateCmd = &cobra.Command{
	Use:   "update",
	Short: "Update validator",
	Long:  `Update validator`,
	Run: func(cmd *cobra.Command, args []string) {
		if updateIPFlag == "" {
			_ = cmd.Help()
			return
		}

		var env []byte
		if updateEnvFlag != "" {
			f, err := os.ReadFile(updateEnvFlag)
			if err != nil {
				fmt.Printf("Failed reading environment file: %s", err)
				os.Exit(1)
			}
			fmt.Printf("Initialize validator with the following environment variables:\n%s\n", string(f))
			var confirm string
			fmt.Print("Confirm [Y/n]: ")
			_, err = fmt.Scanln(&confirm)
			confirm = strings.ToLower(confirm)
			if len(confirm) != 1 || confirm != "y" || err != nil {
				fmt.Println("Operation cancelled.")
				os.Exit(0)
			}
			env = f
		}

		client := &http.Client{
			Timeout: 5 * time.Minute,
		}
		hotkeyPhrase := viper.GetString("validator.hotkey_phrase")
		if len(hotkeyPhrase) == 0 {
			hotkeyPhrase = shared.PromptConfigString("validator.hotkey_phrase")
		}
		kp, err := signature.KeyringPairFromSecret(hotkeyPhrase, 42)
		if err != nil {
			fmt.Println("Failed loading validator hotkey")
			return
		}
		headers, err := boilerplate.GetEpistulaHeaders(kp, kp.Address, env)
		if err != nil {
			fmt.Println("Failed generating epistula headers")
			os.Exit(1)
		}
		cvmIP := strings.TrimPrefix(updateIPFlag, "http://")
		req, err := http.NewRequest("POST", fmt.Sprintf("http://%s:8080/api/vali/update", cvmIP), bytes.NewBuffer(env))
		if err != nil {
			fmt.Printf("Faield sending request to VM: %s", err)
			os.Exit(1)
		}
		for key, value := range headers {
			req.Header.Set(key, value)
		}
		req.Close = true

		resp, err := client.Do(req)
		if err != nil {
			fmt.Printf("Faield sending request to VM: %s", err)
			os.Exit(1)
		}
		defer func() {
			_ = resp.Body.Close()
		}()
		b, err := io.ReadAll(resp.Body)
		if err != nil {
			fmt.Printf("Faield reading response from VM: %s", err)
			os.Exit(1)
		}
		if resp.StatusCode != 200 {
			fmt.Printf("Failed sending request to vm. Got status code %d: %s", resp.StatusCode, string(b))
			os.Exit(1)
		}
		fmt.Println("Validator Updated")
	},
}
