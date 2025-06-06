package monitor

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"miner/internal/setup"
)

func GetAndRegNodes(deps *setup.Dependencies) []string {
	newNodes := []string{}
	for _, node := range deps.Config.Nodes {
		deps.Log.Infof("Checking node registration %s", node)
		tr := &http.Transport{
			TLSHandshakeTimeout: 5 * time.Second,
			MaxConnsPerHost:     1,
			DisableKeepAlives:   true,
		}
		client := &http.Client{Transport: tr, Timeout: 20 * time.Second}
		node = strings.TrimPrefix(node, "http://")
		node = strings.TrimSuffix(node, ":8080")
		ss58 := map[string]string{
			"ss58": deps.Hotkey.Address,
		}
		body, _ := json.Marshal(ss58)
		req, err := http.NewRequest("POST", fmt.Sprintf("http://%s:8080/api/v1/register", node), bytes.NewBuffer(body))
		if err != nil {
			deps.Log.Debugw("Failed to generate request to miner", "error", err)
			continue
		}
		req.Close = true
		resp, err := client.Do(req)
		if err != nil {
			deps.Log.Debugw("Failed sending request to miner", "error", err)
			continue
		}
		respBody, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if strings.TrimSpace(string(respBody)) != deps.Hotkey.Address {
			deps.Log.Errorf("CVM Hotkey is not self hotkey: %s", respBody)
			continue
		}
		newNodes = append(newNodes, node)
	}
	return newNodes
}

func MonitorNodes(deps *setup.Dependencies) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	go func() {
		for range ticker.C {
			deps.Log.Info("Checking nodes")
			for _, n := range deps.Config.Nodes {
				ok := CheckCVMHealth(deps, n)
				if !ok {
					deps.Log.Errorf("Failed checking health of node at ip %s", n)
				}
			}
		}
	}()
}

func CheckCVMHealth(deps *setup.Dependencies, cvmIP string) bool {
	tr := &http.Transport{
		TLSHandshakeTimeout: 5 * time.Second,
		MaxConnsPerHost:     1,
		DisableKeepAlives:   true,
	}
	client := &http.Client{Transport: tr, Timeout: 5 * time.Second}
	cvmIP = strings.TrimPrefix(cvmIP, "http://")
	cvmIP = strings.TrimSuffix(cvmIP, ":8080")
	req, err := http.NewRequest("GET", fmt.Sprintf("http://%s:8080/health", cvmIP), nil)
	if err != nil {
		deps.Log.Debugw("Failed to generate request to miner", "error", err)
		return false
	}
	req.Close = true
	resp, err := client.Do(req)
	if err != nil {
		deps.Log.Debugw("Failed sending request to miner", "error", err)
		return false
	}
	defer func() {
		_ = resp.Body.Close()
	}()
	return resp.StatusCode == http.StatusOK
}
