package callbacks

import (
	"errors"
	"strings"
	"sync"

	"targon/internal/cvm"
	"targon/internal/subtensor/utils"
	"targon/internal/targon"
)

// Collects attestations across all miners and all nodes, skipping nodes
// its already found
func getPassingAttestations(c *targon.Core) {
	wg := sync.WaitGroup{}
	c.Deps.Log.Infof("Getting Attestations for %d miners", len(c.Neurons))
	attester := cvm.NewAttester(c.Deps.Env.TIMEOUT_MULT, c.Deps.Hotkey, c.Deps.Env.NVIDIA_ATTEST_ENDPOINT)

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
			c.Mu.Unlock()

			wg.Add(1)
			go func() {
				defer wg.Done()

				// Get attestation
				n := c.Neurons[uid]
				nonce := targon.NewNonce(attester.Hotkey.Address)
				cvmIP := strings.TrimPrefix(node.Ip, "http://")
				cvmIP = strings.TrimSuffix(cvmIP, ":8080")
				attestPayload, err := attester.GetAttestFromNode(utils.AccountIDToSS58(n.Hotkey), cvmIP, nonce)

				// Verify attestation
				var gpus, ueids []string
				if err == nil {
					gpus, ueids, err = attester.CheckAttest(
						attestPayload,
						nonce,
					)
				}

				// Lock for core map updates
				c.Mu.Lock()
				defer c.Mu.Unlock()

				// Check for duplicate GPUS
				if err == nil {
					for _, v := range ueids {
						if c.GPUids[v] {
							// Add empty string so that we dont ping this node again,
							// but dont pass any actual gpus
							c.PassedAttestation[uid][node.Ip] = []string{}
							err = errors.New("duplicate gpu id found")
							break
						}
						c.GPUids[v] = true
					}
				}

				// Check with tower for this ip
				if err == nil {
					passed := c.Deps.Tower.Check(node.Ip)
					if !passed {
						err = errors.New("failed tower check")
					}
				}

				// Mark error if found; all errors here are non-retryable
				if err != nil {
					c.AttestErrors[uid][node.Ip] = err.Error()
					c.Deps.Log.Debugw("failed attestation", "ip", node.Ip, "uid", uid, "error", err)

					// Check if its a retryable error
					var aerr *cvm.AttestError
					if errors.As(err, &aerr) {
						c.Deps.Log.Debugf("%s: attest error: retry: %t, msg: %s", uid, aerr.ShouldRetry, aerr.Msg)
						if !aerr.ShouldRetry {
							c.PassedAttestation[uid][node.Ip] = []string{}
						}
						return
					}
					return
				}

				// Add gpus to passed gpus, and delete any error marks
				// Only add gpus if not duplicates
				c.Deps.Log.Infof("%s passed attestation: %s", uid, gpus)
				c.PassedAttestation[uid][node.Ip] = gpus
				delete(c.AttestErrors[uid], node.Ip)
			}()
		}
	}
	wg.Wait()
}
