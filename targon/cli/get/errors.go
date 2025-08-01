package get

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"targon/cli/shared"
	"targon/internal/utils"

	"github.com/centrifuge/go-substrate-rpc-client/v4/signature"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"github.com/subtrahend-labs/gobt/boilerplate"
)

var (
	uidflag  int
	endpoint string
)

func init() {
	errsCMD.Flags().IntVar(&uidflag, "uid", -1, "Specific uid to grab GPU info for")
	errsCMD.Flags().StringVar(&endpoint, "endpoint", "https://stats.targon.com/api/miners/attest/error", "Endpoint to get errors from")
	getCmd.AddCommand(errsCMD)
}

type AttestErrors struct {
	Data map[string]string `json:"data"`
}

var errsCMD = &cobra.Command{
	Use:   "errors",
	Short: "Get attestation errors for UID",
	Long:  `Get attestation errors for UID`,
	Run: func(cmd *cobra.Command, args []string) {
		hotkeyPhrase := viper.GetString("miner.hotkey_phrase")
		if len(hotkeyPhrase) == 0 {
			hotkeyPhrase = shared.PromptConfigString("miner.hotkey_phrase")
		}
		kp, err := signature.KeyringPairFromSecret(hotkeyPhrase, 42)
		if err != nil {
			fmt.Println("Failed loading miner hotkey")
			return
		}
		req, err := http.NewRequest("GET", endpoint+"/"+fmt.Sprintf("%d", uidflag), nil)
		if err != nil {
			err := utils.Wrap("Failed to generate request", err)
			fmt.Println(err)
			return
		}

		headers, err := boilerplate.GetEpistulaHeaders(kp, "", []byte{})
		if err != nil {
			err := utils.Wrap("Failed generating epistula headers", err)
			fmt.Println(err)
			return
		}
		for key, value := range headers {
			req.Header.Set(key, value)
		}
		req.Close = true

		tr := &http.Transport{
			TLSHandshakeTimeout: 5 * time.Second,
			MaxConnsPerHost:     1,
			DisableKeepAlives:   true,
		}
		client := &http.Client{Transport: tr, Timeout: 10 * time.Second}
		resp, err := client.Do(req)
		if err != nil {
			err := utils.Wrap("Failed sending request to stats", err)
			fmt.Println(err)
			return
		}
		defer func() {
			_ = resp.Body.Close()
		}()
		if resp.StatusCode != 200 {
			fmt.Printf("Failed sending request to stats: %s\n", resp.Status)
			return
		}
		body, _ := io.ReadAll(resp.Body)
		var attErrs AttestErrors
		_ = json.Unmarshal(body, &attErrs)
		for k, v := range attErrs.Data {
			fmt.Printf("%s: %s\n", k, v)
		}
	},
}
