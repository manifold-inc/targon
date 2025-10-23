package vali

import (
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

var updateIPFlag string

func init() {
	valiCmd.AddCommand(updateCmd)
	updateCmd.Flags().StringVar(&updateIPFlag, "ip", "localhost", "IP address of the vm")
}

var updateCmd = &cobra.Command{
	Use:   "update",
	Short: "Update validator",
	Long:  `Update validator`,
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		if updateIPFlag == "" {
			_ = cmd.Help()
			return
		}
		client := &http.Client{
			Timeout: 10 * time.Second,
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
		headers, err := boilerplate.GetEpistulaHeaders(kp, kp.Address, []byte{})
		if err != nil {
			fmt.Println("Failed generating epistula headers")
			os.Exit(1)
		}
		cvmIP := strings.TrimPrefix(updateIPFlag, "http://")
		cvmIP = strings.TrimSuffix(cvmIP, ":8080/api/vali/update")
		req, err := http.NewRequest("POST", cvmIP, nil)
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
