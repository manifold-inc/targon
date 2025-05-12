package monitor

import (
	"fmt"
	"miner/cmd/internal/setup"
	"net/http"
	"strings"
	"time"
)

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
