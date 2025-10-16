package vali

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"
)

func init() {
	valiCmd.AddCommand(initCmd)
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
			Timeout: 10 * time.Second,
		}

		req, err := http.NewRequest("POST", "http://localhost:8080/api/vali/init", bytes.NewBuffer(f))
		if err != nil {
			fmt.Printf("Faield sending request to VM: %s", err)
			os.Exit(1)
		}
		req.Header.Set("Content-Type", "text/plain")

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
		fmt.Println("Validator started")
	},
}
