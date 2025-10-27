package vali

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/centrifuge/go-substrate-rpc-client/v4/signature"
	"github.com/docker/docker/api/types/container"
	"github.com/manifold-inc/manifold-sdk/lib/utils"
	"github.com/spf13/cobra"
	"github.com/subtrahend-labs/gobt/boilerplate"
)

var containersIPFlag string

func init() {
	valiCmd.AddCommand(containersCMD)
	containersCMD.Flags().StringVar(&containersIPFlag, "ip", "localhost", "IP address of the vm")
}

var containersCMD = &cobra.Command{
	Use:   "containers",
	Short: "Get containers from a vm",
	Long:  `Get containers from a vm`,
	Run: func(cmd *cobra.Command, args []string) {
		if containersIPFlag == "" {
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

		if len(containersIPFlag) != 0 {
			cvmIP := strings.TrimPrefix(containersIPFlag, "http://")
			cvmIP = strings.TrimSuffix(cvmIP, ":8080")

			containers, err := GetContainers(cvmIP, kp)
			if err != nil {
				fmt.Println(utils.Wrap("error getting containers from cvm", err))
				return
			}
			for _, con := range containers {
				name := strings.TrimPrefix(con.Names[0], "/")
				fmt.Println(name)
			}
			return
		}
	},
}

func GetContainers(
	cvmIP string,
	kp signature.KeyringPair,
) ([]container.Summary, error) {
	client := &http.Client{Transport: &http.Transport{
		TLSHandshakeTimeout: 5 * time.Second,
		MaxConnsPerHost:     1,
		DisableKeepAlives:   true,
		Dial: (&net.Dialer{
			Timeout: 15 * time.Second,
		}).Dial,
	}, Timeout: 5 * time.Minute}

	req, err := http.NewRequest(
		"GET",
		fmt.Sprintf("http://%s:8080/api/v1/containers", cvmIP),
		nil,
	)
	if err != nil {
		return nil, utils.Wrap("failed to generate request to cvm", err)
	}

	headers, err := boilerplate.GetEpistulaHeaders(kp, kp.Address, []byte{})
	if err != nil {
		return nil, utils.Wrap("failed generating epistula headers", err)
	}

	req.Header.Set("Content-Type", "application/json")
	for key, value := range headers {
		req.Header.Set(key, value)
	}

	req.Close = true
	res, err := client.Do(req)
	if err != nil {
		return nil, utils.Wrap("failed sending request to cvm", err)
	}
	defer func() {
		_ = res.Body.Close()
	}()

	if res.StatusCode == http.StatusServiceUnavailable {
		return nil, errors.New("server overloaded")
	}
	if res.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(res.Body)
		return nil, fmt.Errorf("bad status code from cvm containers: %d: %s", res.StatusCode, string(body))
	}

	resBody, err := io.ReadAll(res.Body)
	if err != nil {
		return nil, utils.Wrap("failed reading response from cvm", err)
	}

	var containers []container.Summary
	err = json.Unmarshal(resBody, &containers)
	if err != nil {
		return nil, utils.Wrap("failed to unmarshal containers", err)
	}
	return containers, nil
}
