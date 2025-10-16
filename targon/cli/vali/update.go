package vali

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/spf13/cobra"
)

func init() {
	valiCmd.AddCommand(updateCmd)
}

var updateCmd = &cobra.Command{
	Use:   "update",
	Short: "Update validator",
	Long:  `Update validator`,
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		client := &http.Client{
			Timeout: 10 * time.Second,
		}
		req, err := http.NewRequest("POST", "http://localhost:8080/api/vali/update", nil)
		if err != nil {
			fmt.Printf("Faield sending request to VM: %s", err)
			os.Exit(1)
		}

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
