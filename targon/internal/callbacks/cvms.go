package callbacks

import (
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"targon/internal/cvm"
	"targon/internal/targon"
)

// Collects attestations across all miners and all nodes, skipping nodes
// its already found
func getPassingAttestations(c *targon.Core) {
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
		c.Mu.Lock()
		if c.PassedAttestation[uid] == nil {
			c.PassedAttestation[uid] = map[string][]string{}
		}
		if c.AttestErrors[uid] == nil {
			c.AttestErrors[uid] = map[string]string{}
		}
		c.Mu.Unlock()
		for _, node := range nodes {
			// Dont check nodes that have already passed this interval
			c.Mu.Lock()
			if c.PassedAttestation[uid][node.Ip] != nil {
				c.Mu.Unlock()
				continue
			}
			if val, ok := c.AttestErrors[uid][node.Ip]; ok {
				c.Mu.Unlock()
				c.Deps.Log.Infof("%s skipping for reason: %s", uid, val)
				continue
			}
			c.Mu.Unlock()

			wg.Add(1)
			go func() {
				defer wg.Done()
				err := attest(c, uid, node, verifyAttestClient)
				var berr *cvm.BusyError
				if errors.As(err, &berr) {
					c.Deps.Log.Infof("%s overloaded", uid)
					return
				}

				c.Mu.Lock()
				defer c.Mu.Unlock()
				if err == nil {
					delete(c.AttestErrors[uid], node.Ip)
					return
				}
				if c.AttestErrors[uid] == nil {
					c.AttestErrors[uid] = map[string]string{}
				}
				c.AttestErrors[uid][node.Ip] = err.Error()
				c.Deps.Log.Debugw("failed attestation", "ip", node.Ip, "uid", uid, "error", err)
			}()
		}
	}
	wg.Wait()
}

// Gets attestation result from a specific node. no error means passing node.
func attest(c *targon.Core, uid string, node *targon.MinerNode, attestClient *http.Client) error {
	n := c.Neurons[uid]
	nonce := targon.NewNonce(c.Deps.Hotkey.Address)
	cvmIP := strings.TrimPrefix(node.Ip, "http://")
	cvmIP = strings.TrimSuffix(cvmIP, ":8080")
	log := c.Deps.Log.With("uid", uid, "ip", cvmIP)
	attestPayload, err := cvm.GetAttestFromNode(c.Deps.Hotkey, c.Deps.Env.TIMEOUT_MULT, &n, cvmIP, nonce)
	if err != nil {
		return err
	}
	gpus, ueids, err := cvm.CheckAttest(
		c.Deps.Env.NVIDIA_ATTEST_ENDPOINT,
		attestClient,
		attestPayload.Attest,
		nonce,
	)
	if err != nil {
		return err
	}

	passed := c.Deps.Tower.Check(cvmIP)
	if !passed {
		return errors.New("failed tower check")
	}
	log.Infow("Node successfully verified",
		"gpu_types", fmt.Sprintf("%v", gpus),
	)

	// ensure no duplicate nodes
	c.Mu.Lock()
	defer c.Mu.Unlock()
	for _, v := range ueids {
		if c.GPUids[v] {
			// Add empty string so that we dont ping this node again,
			// but dont pass any actual gpus
			c.PassedAttestation[uid][node.Ip] = []string{}
			return errors.New("duplicate gpu id found")
		}
		c.GPUids[v] = true
	}
	// Only add gpus if not duplicates
	c.PassedAttestation[uid][node.Ip] = gpus
	return nil
}
