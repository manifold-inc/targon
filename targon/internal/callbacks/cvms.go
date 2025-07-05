package callbacks

import (
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"targon/internal/cvm"
	"targon/internal/targon"
)

func getPassingAttestations(c *targon.Core) {
	attestClient := &http.Client{Transport: &http.Transport{
		TLSHandshakeTimeout: 5 * time.Second * c.Deps.Env.TIMEOUT_MULT,
		MaxConnsPerHost:     1,
		DisableKeepAlives:   true,
	}, Timeout: 5 * time.Minute * c.Deps.Env.TIMEOUT_MULT}

	verifyAttestClient := &http.Client{Transport: &http.Transport{
		TLSHandshakeTimeout: 5 * time.Second * c.Deps.Env.TIMEOUT_MULT,
		DisableKeepAlives:   true,
	}, Timeout: 3 * time.Minute * c.Deps.Env.TIMEOUT_MULT}
	wg := sync.WaitGroup{}
	c.Deps.Log.Infof("Getting Attestations for %d miners", len(c.Neurons))

	for uid, nodes := range c.MinerNodes {
		if nodes == nil {
			continue
		}
		for _, node := range nodes {
			// Dont check nodes that have already passed this interval
			if c.PassedAttestation[uid] == nil {
				c.Mu.Lock()
				c.PassedAttestation[uid] = map[string][]string{}
				c.ICONS[uid] = map[string]string{}
				c.Mu.Unlock()
			}
			if c.PassedAttestation[uid][node] != nil {
				continue
			}
			wg.Add(1)
			go func() {
				defer wg.Done()
				n := c.Neurons[uid]
				uid := fmt.Sprintf("%d", n.UID.Int64())
				nonce := targon.NewNonce(c.Deps.Hotkey.Address)
				cvmIP := strings.TrimPrefix(node, "http://")
				cvmIP = strings.TrimSuffix(cvmIP, ":8080")
				log := c.Deps.Log.With("uid", uid, "ip", cvmIP)
				attestPayload, err := cvm.GetAttestFromNode(log, c, attestClient, &n, cvmIP, nonce)
				if err != nil {
					return
				}
				icon := attestPayload.ICON
				gpus, ueids, err := cvm.CheckAttest(
					log,
					c,
					verifyAttestClient,
					attestPayload.Attest,
					nonce,
				)
				if err != nil {
					return
				}

				passed := c.Deps.Tower.Check(cvmIP)
				if !passed {
					log.Info("Node failed tower check")
					return
				}
				log.Info("Node successfully verified")

				// ensure no duplicate nodes
				c.Mu.Lock()
				defer c.Mu.Unlock()
				c.ICONS[uid][node] = icon
				for _, v := range ueids {
					if c.GPUids[v] {
						c.Deps.Log.Infow("Found duplicate GPU ID", "uid", uid)
						// Add empty string so that we dont ping this node again,
						// but dont pass any actual gpus
						c.PassedAttestation[uid][node] = []string{}
						return
					}
					c.GPUids[v] = true
				}
				// Only add gpus if not duplicates
				c.PassedAttestation[uid][node] = gpus
			}()
		}
	}

	// TODO make this more robust, dont want to blast cvm IPs though
	var beersData []targon.GPUData
	for uid, nodes := range c.PassedAttestation {
		if nodes == nil {
			continue
		}
		ns := []string{}
		for _, gpus := range nodes {
			ns = append(ns, gpus...)
		}
		beersData = append(
			beersData,
			targon.GPUData{Uid: uid, Data: map[string][]string{"nodes": ns}},
		)
	}
	client := &http.Client{Transport: &http.Transport{
		TLSHandshakeTimeout: 5 * time.Second * c.Deps.Env.TIMEOUT_MULT,
		MaxConnsPerHost:     1,
		DisableKeepAlives:   true,
	}, Timeout: 1 * time.Minute * c.Deps.Env.TIMEOUT_MULT}
	if err := targon.SendGPUDataToBeers(c, client, beersData); err != nil {
		c.Deps.Log.Warnw("Failed to send GPU data to beers", "error", err)
	}
	wg.Wait()
}

func pingHealthChecks(c *targon.Core) {
	tr := &http.Transport{
		TLSHandshakeTimeout: 5 * time.Second,
		MaxConnsPerHost:     1,
		DisableKeepAlives:   true,
	}
	client := &http.Client{Transport: tr, Timeout: 5 * time.Second}
	wg := sync.WaitGroup{}
	c.Deps.Log.Info("Pinging healthchecks")
	for uid, nodes := range c.PassedAttestation {
		if nodes == nil {
			continue
		}
		for node := range nodes {
			// Dont check nodes that have already passed this interval
			wg.Add(1)
			go func() {
				defer wg.Done()
				c.Mu.Lock()
				n := c.Neurons[uid]
				c.Mu.Unlock()
				ok := cvm.CheckHealth(c, client, &n, node)
				c.Hpmu.Lock()
				defer c.Hpmu.Unlock()
				if c.HealthcheckPasses[uid] == nil {
					c.HealthcheckPasses[uid] = map[string][]bool{}
				}
				if c.HealthcheckPasses[uid][node] == nil {
					c.HealthcheckPasses[uid][node] = []bool{}
				}
				c.HealthcheckPasses[uid][node] = append(c.HealthcheckPasses[uid][node], ok)
			}()
		}
	}
	wg.Wait()
}
