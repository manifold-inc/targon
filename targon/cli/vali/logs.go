package vali

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"strings"
	"time"

	"targon/cli/shared"

	"github.com/centrifuge/go-substrate-rpc-client/v4/signature"
	"github.com/manifold-inc/manifold-sdk/lib/utils"
	"github.com/subtrahend-labs/gobt/boilerplate"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var (
	logsIPFlag        string
	logsContainerFlag string
	logsTailFlag      string
)

func init() {
	valiCmd.AddCommand(logsCMD)
	logsCMD.Flags().StringVar(&logsIPFlag, "ip", "localhost", "IP address of the vm")
	logsCMD.Flags().StringVar(&logsContainerFlag, "container", "", "Name of the container to get logs from")
	logsCMD.Flags().StringVar(&logsTailFlag, "tail", "all", "Number of lines to show from the end of the logs")
}

var logsCMD = &cobra.Command{
	Use:   "logs",
	Short: "View logs for a vm",
	Long:  `View logs for a vm`,
	Run: func(cmd *cobra.Command, args []string) {
		if logsIPFlag == "" {
			_ = cmd.Help()
			return
		}

		config, err := loadConfig()
		if err != nil {
			fmt.Println("error loading config: " + err.Error())
			os.Exit(1)
		}

		kp, err := signature.KeyringPairFromSecret(config.ValidatorHotkeyPhrase, 42)
		if err != nil {
			fmt.Println("error parsing hotkey phrase: " + err.Error())
			os.Exit(1)
		}

		cvmIP := strings.TrimPrefix(logsIPFlag, "http://")
		cvmIP = strings.TrimSuffix(cvmIP, ":8080")

		logs, err := GetLogsFromNode(cvmIP, logsContainerFlag, logsTailFlag, kp)
		if err != nil {
			fmt.Println(utils.Wrap("error getting logs from cvm", err))
			return
		}
		fmt.Printf("Logs:\n%s\n", logs)
	},
}

type GetConfig struct {
	ValidatorHotkeyPhrase string
}

func loadConfig() (*GetConfig, error) {
	config := &GetConfig{}

	configStrings := map[string]*string{
		"validator.hotkey_phrase": &config.ValidatorHotkeyPhrase,
	}

	for key, value := range configStrings {
		if viper.GetString(key) == "" {
			shared.PromptConfigString(key)
		}
		*value = viper.GetString(key)
	}

	return config, nil
}

func GetLogsFromNode(
	cvmIP string,
	containerName string,
	tail string,
	kp signature.KeyringPair,
) (string, error) {
	client := &http.Client{Transport: &http.Transport{
		TLSHandshakeTimeout: 5 * time.Second,
		MaxConnsPerHost:     1,
		DisableKeepAlives:   true,
		DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			dial := &net.Dialer{
				Timeout: 15 * time.Second,
			}
			return dial.DialContext(ctx, "tcp4", addr)
		},
	}, Timeout: 5 * time.Minute}

	data := LogsBody{
		ContainerName: containerName,
		Tail:          tail,
	}
	body, _ := json.Marshal(data)

	req, err := http.NewRequest(
		"POST",
		fmt.Sprintf("http://%s:8080/api/v1/logs", cvmIP),
		bytes.NewBuffer(body),
	)
	if err != nil {
		return "", utils.Wrap("failed to generate request to cvm", err)
	}

	headers, err := boilerplate.GetEpistulaHeaders(kp, kp.Address, body)
	if err != nil {
		return "", utils.Wrap("failed generating epistula headers", err)
	}

	req.Header.Set("Content-Type", "application/json")
	for key, value := range headers {
		req.Header.Set(key, value)
	}

	req.Close = true
	res, err := client.Do(req)
	if err != nil {
		return "", utils.Wrap("failed sending request to cvm", err)
	}
	defer func() {
		_ = res.Body.Close()
	}()

	if res.StatusCode == http.StatusServiceUnavailable {
		return "", errors.New("server overloaded")
	}
	if res.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(res.Body)
		return "", fmt.Errorf("bad status code from cvm logs: %d: %s", res.StatusCode, string(body))
	}

	resBody, err := io.ReadAll(res.Body)
	if err != nil {
		return "", utils.Wrap("failed reading response from cvm", err)
	}
	return string(resBody), nil
}

type LogsBody struct {
	ContainerName string `json:"container_name"`
	Tail          string `json:"tail"`
}
