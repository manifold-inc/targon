package targon

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"targon/internal/discord"
	"targon/internal/setup"

	"github.com/centrifuge/go-substrate-rpc-client/v4/types"
	"github.com/subtrahend-labs/gobt/boilerplate"
	"github.com/subtrahend-labs/gobt/extrinsics"
	"github.com/subtrahend-labs/gobt/runtime"
	"github.com/subtrahend-labs/gobt/sigtools"
	"github.com/subtrahend-labs/gobt/storage"
)

// TODO
// Confrim set weight hash success

func AddBlockCallbacks(v *boilerplate.BaseChainSubscriber, c *Core) {
	// block timer
	t := time.AfterFunc(1*time.Hour, func() {
		c.Deps.Log.Error("havint seen any blocks in over an hour, am i stuck?")
	})
	v.AddBlockCallback(func(h types.Header) {
		t.Reset(1 * time.Hour)
	})
	v.AddBlockCallback(func(h types.Header) {
		go logBlockCallback(c, h)
	})
	// get neurons
	v.AddBlockCallback(func(h types.Header) {
		// Run after first block of interval
		if h.Number%360 != 1 && len(c.Neurons) != 0 {
			return
		}
		getNeuronsCallback(v, c, h)
	})
	// get emission for this interval
	v.AddBlockCallback(func(h types.Header) {
		if c.EmissionPool != nil {
			return
		}
		taoPrice, err := GetTaoPrice()
		if err != nil {
			c.Deps.Log.Errorw("Failed getting tao price", "error", err)
			return
		}
		c.TaoPrice = &taoPrice
		c.Deps.Log.Infof("Current tao price $%f", *c.TaoPrice)
		p, err := storage.GetSubnetTaoInEmission(c.Deps.Client, types.NewU16(uint16(c.Deps.Env.NETUID)), &h.ParentHash)
		if err != nil {
			c.Deps.Log.Errorw("Failed getting sn tao emissions", "error", err)
			return
		}
		emi := (float64(*p) / 1e9) * .41 * 360 * *c.TaoPrice
		c.EmissionPool = &emi
		c.Deps.Log.Infof("Current sn miner emission pool in $ %f", *c.EmissionPool)
	})
	// get miner nodes
	v.AddBlockCallback(func(h types.Header) {
		if h.Number%360 != 1 && len(c.MinerNodes) != 0 {
			return
		}
		getMinerNodes(c)
	})
	// get passing attestations
	v.AddBlockCallback(func(h types.Header) {
		if c.Neurons == nil {
			return
		}
		blocksTill := 360 - (h.Number % 360)
		if blocksTill < 20 {
			return
		}
		// Not on specific tempo;
		// helps reduce stress on cvm nodes from number of pings
		chance := rand.Float32()
		if chance < .95 && len(c.PassedAttestation) != 0 {
			return
		}
		getPassingAttestations(c)
	})
	// Ping miner healthchecks
	v.AddBlockCallback(func(h types.Header) {
		if h.Number%10 != 0 || len(c.MinerNodes) == 0 {
			return
		}
		pingHealthChecks(c)
	})
	// Log weights
	v.AddBlockCallback(func(h types.Header) {
		if h.Number%10 != 0 || len(c.MinerNodes) == 0 {
			return
		}
		logWeights(c)
	})
	// Discord Notifications
	v.AddBlockCallback(func(h types.Header) {
		if h.Number%720 != 0 {
			return
		}
		sendDailyGPUSummary(c, h)
	})
	// Set Weights
	v.AddBlockCallback(func(h types.Header) {
		if h.Number%360 != 0 || len(c.MinerNodes) == 0 {
			return
		}
		setWeights(v, c, h)
	})
}

func logWeights(c *Core) {
	uids, scores, _ := getWeights(c)
	c.Deps.Log.Infow(
		"Current Weights",
		"uids",
		fmt.Sprintf("%+v", uids),
		"scores",
		fmt.Sprintf("%+v", scores),
	)
}

func logBlockCallback(c *Core, h types.Header) {
	// Run Every Block
	c.Deps.Log.Infow(
		"New block",
		"block",
		fmt.Sprintf("%v", h.Number),
		"left_in_interval",
		fmt.Sprintf("%d", 360-(h.Number%360)),
	)
}

func getNeuronsCallback(v *boilerplate.BaseChainSubscriber, c *Core, h types.Header) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.Deps.Log.Info("Updating neurons")
	blockHash, err := c.Deps.Client.Api.RPC.Chain.GetBlockHash(uint64(h.Number))
	if err != nil {
		c.Deps.Log.Errorw("Failed getting blockhash for neurons", "error", err)
		return
	}
	neurons, err := runtime.GetNeurons(c.Deps.Client, uint16(v.NetUID), &blockHash)
	if err != nil {
		c.Deps.Log.Errorw("Failed getting neurons", "error", err)
		return
	}
	for _, n := range neurons {
		uid := fmt.Sprintf("%d", n.UID.Int64())
		c.Neurons[uid] = n
	}
	c.Deps.Log.Info("Neurons Updated")
}

func getMinerNodes(c *Core) {
	tr := &http.Transport{
		TLSHandshakeTimeout: 5 * time.Second,
		MaxConnsPerHost:     1,
		DisableKeepAlives:   true,
	}
	client := &http.Client{Transport: tr, Timeout: 5 * time.Second}
	wg := sync.WaitGroup{}
	wg.Add(len(c.Neurons))
	c.Deps.Log.Infof("Checking CVM nodes for %d miners", len(c.Neurons))
	totalNodes := 0
	for _, n := range c.Neurons {
		go func() {
			uid := fmt.Sprintf("%d", n.UID.Int64())
			defer wg.Done()
			nodes, err := GetCVMNodes(c, client, &n)
			// Either nil or list or strings,
			// errors are logged internally, dont need to handle them here
			c.Mnmu.Lock()
			defer c.Mnmu.Unlock()
			c.MinerNodes[uid] = nodes
			if err != nil {
				return
			}
			totalNodes += len(nodes)
		}()
	}
	wg.Wait()
	c.Deps.Log.Infof("Found %d miners with a total of %d nodes", len(c.MinerNodes), totalNodes)
}

func getPassingAttestations(c *Core) {
	tr := &http.Transport{
		TLSHandshakeTimeout: 5 * time.Second,
		MaxConnsPerHost:     1,
		DisableKeepAlives:   true,
	}
	client := &http.Client{Transport: tr, Timeout: 5 * time.Minute}
	wg := sync.WaitGroup{}
	c.Deps.Log.Infof("Getting Attestations for %d miners", len(c.Neurons))

	for uid, nodes := range c.MinerNodes {
		if nodes == nil {
			continue
		}
		for _, node := range nodes {
			// Dont check nodes that have already passed this interval
			if c.PassedAttestation[uid] == nil {
				c.mu.Lock()
				c.PassedAttestation[uid] = map[string][]string{}
				c.ICONS[uid] = map[string]string{}
				c.mu.Unlock()
			}
			if c.PassedAttestation[uid][node] != nil {
				continue
			}
			wg.Add(1)
			go func() {
				defer wg.Done()
				n := c.Neurons[uid]
				gpus, ueids, icon, err := CheckCVMAttest(c, client, &n, node)
				if err != nil {
					return
				}

				// ensure no duplicate nodes
				c.mu.Lock()
				defer c.mu.Unlock()
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
	var beersData []GPUData
	for uid, nodes := range c.PassedAttestation {
		if nodes == nil {
			continue
		}
		ns := []string{}
		for _, gpus := range nodes {
			ns = append(ns, gpus...)
		}
		beersData = append(beersData, GPUData{Uid: uid, Data: map[string][]string{"nodes": ns}})
	}
	if err := sendGPUDataToBeers(c, client, beersData); err != nil {
		c.Deps.Log.Warnw("Failed to send GPU data to beers", "error", err)
	}
	wg.Wait()
}

func pingHealthChecks(c *Core) {
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
				c.mu.Lock()
				n := c.Neurons[uid]
				c.mu.Unlock()
				ok := CheckCVMHealth(c, client, &n, node)
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

func resetState(c *Core) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.Neurons = make(map[string]runtime.NeuronInfo)
	c.MinerNodes = make(map[string][]string)
	c.GPUids = make(map[string]bool)
	// Dont really need to wipe tao price
	c.EmissionPool = nil
	// TODO maybe keep this alive longer than an interval
	c.HealthcheckPasses = make(map[string]map[string][]bool)
	c.PassedAttestation = make(map[string]map[string][]string)
	c.ICONS = make(map[string]map[string]string)
}

func setWeights(v *boilerplate.BaseChainSubscriber, c *Core, h types.Header) {
	c.mu.Lock()
	defer func() {
		c.mu.Unlock()
		resetState(c)
	}()
	uids, scores, err := getWeights(c)
	if err != nil {
		c.Deps.Log.Errorw("Failed getting weights", "error", err)
	}
	c.Deps.Log.Infow(
		"Setting Weights",
		"uids",
		fmt.Sprintf("%+v", uids),
		"scores",
		fmt.Sprintf("%+v", scores),
	)

	// Store emissions data to MongoDB
	if c.Deps.Mongo != nil {
		uidsUint16 := make([]uint16, len(uids))
		scoresUint16 := make([]uint16, len(scores))
		rawScores := make([]float64, len(scores))

		for i, uid := range uids {
			uidsUint16[i] = uint16(uid)
		}
		for i, score := range scores {
			scoresUint16[i] = uint16(score)
			rawScores[i] = (float64(score) / float64(setup.U16MAX)) * *c.EmissionPool
		}

		minerInfo := MinerInfo{
			Core:         c,
			Block:        int(h.Number),
			UIDs:         uidsUint16,
			Scores:       scoresUint16,
			RawScores:    rawScores,
			MinerScores:  nil,
			EmissionPool: c.EmissionPool,
			TaoPrice:     c.TaoPrice,
			Timestamp:    time.Now().Unix(),
		}

		if err := SyncMongo(c, minerInfo); err != nil {
			c.Deps.Log.Errorw("Failed syncing to mongo", "error", err)
		} else {
			c.Deps.Log.Infow("Stored miner info to MongoDB", "block", h.Number)
		}
	}

	go func() {
		color := "3447003"
		title := fmt.Sprintf("Validator setting weights at block %v", h.Number)
		desc := fmt.Sprintf("UIDS: %v\n\nweights: %v", uids, scores)
		uname := "Validator Logs"
		msg := discord.Message{
			Username: &uname,
			Embeds: &[]discord.Embed{{
				Title:       &title,
				Description: &desc,
				Color:       &color,
			}},
		}
		err := discord.SendDiscordMessage(c.Deps.Env.DISCORD_URL, msg)
		if err != nil {
			c.Deps.Log.Warnw("Failed sending discord webhook", "error", err)
		}
	}()
	if c.Deps.Env.DEBUG {
		c.Deps.Log.Warn("Skipping weightset due to debug flag")
		return
	}
	// Actually set weights
	ext, err := extrinsics.SetWeightsExt(
		c.Deps.Client,
		types.U16(v.NetUID),
		uids,
		scores,
		c.Deps.Env.VERSION,
	)
	if err != nil {
		c.Deps.Log.Warnw("Failed creating setweights ext", "error", err)
		return
	}
	ops, err := sigtools.CreateSigningOptions(c.Deps.Client, c.Deps.Hotkey, nil)
	if err != nil {
		c.Deps.Log.Errorw("Failed creating sigining opts", "error", err)
		return
	}
	err = ext.Sign(
		c.Deps.Hotkey,
		c.Deps.Client.Meta,
		ops...,
	)
	if err != nil {
		c.Deps.Log.Errorw("Error signing setweights", "error", err)
		return
	}

	hash, err := c.Deps.Client.Api.RPC.Author.SubmitExtrinsic(*ext)
	if err != nil {
		c.Deps.Log.Errorw("Error submitting extrinsic", "error", err)
		return
	}
	c.Deps.Log.Infow("Set weights on chain successfully", "hash", hash.Hex())
}

func getWeights(c *Core) ([]types.U16, []types.U16, error) {
	if c.EmissionPool == nil {
		return []types.U16{}, []types.U16{}, errors.New("emission pool is not set")
	}
	minerCut := 0.0
	var uids []types.U16
	var scores []float64
	var cvmNodes []string
	gpus := map[string]int{}
	// for each uid
	for uid, nodes := range c.MinerNodes {
		thisScore := 0.0
		// for each node
		for _, n := range nodes {
			if c.PassedAttestation[uid] == nil {
				continue
			}
			if c.PassedAttestation[uid][n] == nil {
				continue
			}
			cvmNodes = append(cvmNodes, n)
			// for each gpu
			for _, gpu := range c.PassedAttestation[uid][n] {
				ml := strings.ToLower(gpu)
				gpus[ml] += 1
				// score is GPU cost per hour * hours in interval (1.5) / total coming
				// into sn this interval
				switch {
				case strings.Contains(ml, "h100"):
					score := (2.5 * 1.233) / *c.EmissionPool
					thisScore += score
					minerCut += score
				case strings.Contains(ml, "h200"):
					score := (3.5 * 1.233) / *c.EmissionPool
					thisScore += score
					minerCut += score
				default:
					continue
				}
			}
		}
		if thisScore < 0.01 {
			continue
		}
		uidInt, _ := strconv.Atoi(uid)

		uids = append(uids, types.NewU16(uint16(uidInt)))
		scores = append(scores, thisScore)
	}
	burnKey := 28
	minerCut = math.Min(minerCut, 1) // Diulte after 100% emission hit
	scores = Normalize(scores, minerCut)
	scores = append(scores, 1-minerCut)
	uids = append(uids, types.NewU16(uint16(burnKey)))

	for gpu, count := range gpus {
		c.Deps.Log.Infof("%s count: %d", gpu, count)
	}
	c.Deps.Log.Infof("CVM IPs: %v", cvmNodes)
	c.Deps.Log.Infow("Miner scores", "uids", fmt.Sprintf("%v", uids), "scores", fmt.Sprintf("%v", scores))

	var finalScores []types.U16
	var finalUids []types.U16
	sumScores := uint16(0)
	for i, s := range scores {
		// send dust to burn
		if i == len(scores)-1 {
			continue
		}
		fw := math.Floor(float64(setup.U16MAX) * s)
		if fw == 0 {
			continue
		}
		thisScore := uint16(fw)
		finalScores = append(finalScores, types.NewU16(thisScore))
		finalUids = append(finalUids, uids[i])
		sumScores += thisScore
	}
	finalScores = append(finalScores, types.NewU16(setup.U16MAX-sumScores))
	finalUids = append(finalUids, types.NewU16(uint16(burnKey)))

	return finalUids, finalScores, nil
}

type minerStats struct {
	gpuCount int
	gpuTypes map[string]int
}

func sendDailyGPUSummary(c *Core, h types.Header) {
	stats := make(map[string]*minerStats)
	totalGPUs := 0
	activeNodes := 0

	for uid, nodes := range c.PassedAttestation {
		if stats[uid] == nil {
			stats[uid] = &minerStats{
				gpuTypes: make(map[string]int),
			}
		}

		for _, gpus := range nodes {
			activeNodes++
			for _, gpu := range gpus {
				gpuLower := strings.ToLower(gpu)
				totalGPUs++
				stats[uid].gpuCount++
				stats[uid].gpuTypes[gpuLower]++
			}
		}
	}

	// Aggregate GPU types across all miners
	gpuTypes := make(map[string]int)
	for _, miner := range stats {
		for gpu, count := range miner.gpuTypes {
			gpuTypes[gpu] += count
		}
	}

	color := "5763719"
	title := fmt.Sprintf("Daily GPU Summary at block %v", h.Number)
	desc := fmt.Sprintf(
		"Total Attested GPUs: %d\n"+
			"Active CVM Nodes: %d\n"+
			"GPU Type Breakdown:\n%s\n"+
			"Per Miner Breakdown:\n%s",
		totalGPUs,
		activeNodes,
		formatGPUBreakdown(gpuTypes),
		formatMinerBreakdown(stats),
	)

	uname := "GPU Monitor"
	msg := discord.Message{
		Username: &uname,
		Embeds: &[]discord.Embed{{
			Title:       &title,
			Description: &desc,
			Color:       &color,
		}},
	}
	err := discord.SendDiscordMessage(c.Deps.Env.DISCORD_URL, msg)
	c.Deps.Log.Infow("Sent daily GPU summary",
		"total_gpus", totalGPUs,
		"active_nodes", activeNodes,
		"gpu_types", gpuTypes,
	)
	if err != nil {
		c.Deps.Log.Warnw("Failed sending discord webhook", "error", err)
	}
}

func formatGPUBreakdown(gpuTypes map[string]int) string {
	var sb strings.Builder
	for gpu, count := range gpuTypes {
		sb.WriteString(fmt.Sprintf("- %s: %d\n", gpu, count))
	}
	return sb.String()
}

func formatMinerBreakdown(stats map[string]*minerStats) string {
	var sb strings.Builder
	for uid, miner := range stats {
		sb.WriteString(fmt.Sprintf("- Miner %s: %d GPUs\n", uid, miner.gpuCount))
	}
	return sb.String()
}
