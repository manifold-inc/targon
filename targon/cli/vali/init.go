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

var initIPFlag string

func init() {
	valiCmd.AddCommand(initCmd)
	initCmd.Flags().StringVar(&initIPFlag, "ip", "localhost", "IP address of the vm")
}

var initCmd = &cobra.Command{
	Use:   "init FILE",
	Short: "Initialize validator",
	Long:  `Initialize validator environment variables and start the validator`,
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		if len(args) == 0 {
			_ = cmd.Help()
			os.Exit(0)
		}
		if initIPFlag == "" {
			_ = cmd.Help()
			return
		}
		f, err := os.ReadFile(args[0])
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
		client := &http.Client{
			Timeout: 5 * time.Minute,
		}

		hotkeyPhrase := viper.GetString("miner.hotkey_phrase")
		if len(hotkeyPhrase) == 0 {
			hotkeyPhrase = shared.PromptConfigString("miner.hotkey_phrase")
		}
		kp, err := signature.KeyringPairFromSecret(hotkeyPhrase, 42)
		if err != nil {
			fmt.Println("Failed loading miner hotkey")
			return
		}
		headers, err := boilerplate.GetEpistulaHeaders(kp, kp.Address, f)
		if err != nil {
			fmt.Println("Failed generating epistula headers")
			os.Exit(1)
		}

		cvmIP := strings.TrimPrefix(initIPFlag, "http://")
		req, err := http.NewRequest("POST", fmt.Sprintf("http://%s:8080/api/vali/init", cvmIP), bytes.NewBuffer(f))
		if err != nil {
			fmt.Printf("Failed sending request to VM: %s", err)
			os.Exit(1)
		}
		req.Header.Set("Content-Type", "text/plain")
		for key, value := range headers {
			req.Header.Set(key, value)
		}
		req.Close = true

		resp, err := client.Do(req)
		if err != nil {
			fmt.Printf("Failed sending request to VM: %s", err)
			os.Exit(1)
		}
		defer func() {
			_ = resp.Body.Close()
		}()
		b, err := io.ReadAll(resp.Body)
		if err != nil {
			fmt.Printf("Failed reading response from VM: %s", err)
			os.Exit(1)
		}
		if resp.StatusCode != 200 {
			fmt.Printf("Failed sending request to vm. Got status code %d: %s", resp.StatusCode, string(b))
			os.Exit(1)
		}
		fmt.Println("Validator started")
	},
}
