package get

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"targon/internal/utils"

	"github.com/centrifuge/go-substrate-rpc-client/v4/signature"

	"github.com/spf13/cobra"
	"github.com/subtrahend-labs/gobt/boilerplate"
)

var uidflag int

func init() {
	ipsCMD.Flags().IntVar(&uidflag, "uid", -1, "Specific uid to grab GPU info for")
	getCmd.AddCommand(ipsCMD)
}

type AttestErrors struct {
	Data map[string]string `json:"data"`
}

var ipsCMD = &cobra.Command{
	Use:   "errors",
	Short: "Get attestation errors for UID",
	Long:  `Get attestation errors for UID`,
	Run: func(cmd *cobra.Command, args []string) {
		// TODO Use viper conifg
		secret := os.Getenv("HOTKEY_PHRASE")
		if len(secret) == 0 {
			fmt.Println("Failed loading miner hotkey, missing env variable `HOTKEY_PHRASE`")
			return
		}
		kp, err := signature.KeyringPairFromSecret(secret, 42)
		if err != nil {
			fmt.Println("Failed loading miner hotkey")
			return
		}
		req, err := http.NewRequest("GET", "https://stats.targon.com/api/miners/attest/error/"+fmt.Sprintf("%d", uidflag), nil)
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
