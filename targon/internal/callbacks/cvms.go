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
	mu := sync.Mutex{}
	c.Deps.Log.Infof("Getting Attestations for %d miners", len(c.Neurons))
	attester := cvm.NewAttester(c.Deps.Env.TIMEOUT_MULT, c.Deps.Hotkey, c.Deps.Env.NVIDIA_ATTEST_ENDPOINT)

	for uid, nodes := range c.MinerNodes {
		if nodes == nil {
			continue
		}
		mu.Lock()
		if c.VerifiedNodes[uid] == nil {
			c.VerifiedNodes[uid] = map[string]*targon.UserData{}
		}
		if c.VerifiedNodes[uid] == nil {
			c.VerifiedNodes[uid] = map[string]*targon.UserData{}
		}
		mu.Unlock()

		for _, node := range nodes {
			// Dont check nodes that have already passed this interval
			mu.Lock()
			if c.VerifiedNodes[uid][node.Ip] != nil {
				mu.Unlock()
				continue
			}
			mu.Unlock()

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
				var gpus, nodeids []string
				if err == nil {
					gpus, nodeids, err = attester.CheckAttest(
						attestPayload,
						nonce,
					)
				}

				// Check with tower for this ip
				if err == nil {
					passed := c.Deps.Tower.Check(node.Ip)
					if !passed {
						err = errors.New("failed tower check")
					}
				}

				// Lock for core map updates
				mu.Lock()
				defer mu.Unlock()

				// Check for duplicate GPUS
				if err == nil {
					for _, v := range nodeids {
						if c.NodeIds[v] {
							err = errors.New("duplicate node id found")
							break
						}
						c.NodeIds[v] = true
					}
				}

				// Mark error if found; all errors here are non-retryable
				if err != nil {
					c.MinerErrors[uid][node.Ip] = err.Error()
					c.Deps.Log.Debugw("failed attestation", "ip", node.Ip, "uid", uid, "error", err.Error())

					// Check if its a retryable error
					var aerr *cvm.AttestError
					if errors.As(err, &aerr) {
						c.Deps.Log.Debugf("%s: attest error: %s", uid, aerr.Error())
						return
					}
					return
				}

				// Add gpus to passed gpus, and delete any error marks
				// Only add gpus if not duplicates
				c.Deps.Log.Infof("%s passed attestation: %s", uid, gpus)

				// TODO add user data
				c.VerifiedNodes[uid][node.Ip] = &targon.UserData{}
				delete(c.MinerErrors[uid], node.Ip)
			}()
		}
	}
	wg.Wait()
}
