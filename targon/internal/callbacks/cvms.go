package callbacks

import (
	"errors"
	"strings"
	"sync"

	"targon/internal/cvm"
	"targon/internal/targon"

	"github.com/manifold-inc/manifold-sdk/lib/utils"
)

// Collects attestations across all miners and all nodes, skipping nodes
// its already found
func getPassingAttestations(c *targon.Core) {
	wg := sync.WaitGroup{}
	mu := sync.Mutex{}
	c.Deps.Log.Infof("Getting Attestations for %d miners", len(c.Neurons))
	attester := cvm.NewAttester(c.Deps.Env.TIMEOUT_MULT, c.Deps.Hotkey, c.Deps.Env.TOWER_URL)

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
		if c.MinerErrors[uid] == nil {
			c.MinerErrors[uid] = make(map[string]string)
		}
		mu.Unlock()

		for _, node := range nodes {
			// Dont check nodes that have already passed this interval
			mu.Lock()
			if c.VerifiedNodes[uid][node.IP] != nil {
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
				cvmIP := strings.TrimPrefix(node.IP, "http://")
				cvmIP = strings.TrimSuffix(cvmIP, ":8080")
				attestPayload, err := attester.GetAttestFromNode(utils.AccountIDToSS58(n.Hotkey), cvmIP, nonce)

				// Verify attestation
				var userData *targon.UserData
				if err == nil {
					userData, err = attester.VerifyAttestation(
						attestPayload,
						nonce,
						node.IP,
					)
				}

				// Lock for core map updates
				mu.Lock()
				defer mu.Unlock()

				// Check for duplicate GPUS
				if err == nil {
					if c.NodeIds[userData.CVMID] {
						err = errors.New("duplicate node id found")
					}
					c.NodeIds[userData.CVMID] = true
				}

				// Mark error if found; all errors here are non-retryable
				if err != nil {
					c.MinerErrors[uid][node.IP] = err.Error()
					c.Deps.Log.Debugw("failed attestation", "ip", node.IP, "uid", uid, "error", err.Error())
					return
				}

				// Add gpus to passed gpus, and delete any error marks
				// Only add gpus if not duplicates
				c.Deps.Log.Infof("%s passed attestation: %s", uid, userData.AuctionName)
				c.VerifiedNodes[uid][node.IP] = userData
				delete(c.MinerErrors[uid], node.IP)
			}()
		}
	}
	wg.Wait()
}
