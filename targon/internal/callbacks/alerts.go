package callbacks

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"targon/internal/discord"
	"targon/internal/setup"
	"targon/internal/targon"

	"github.com/centrifuge/go-substrate-rpc-client/v4/types"
)

type minerStats struct {
	gpuCount  int
	gpuTypes  map[string]int
	uid       int
	incentive float64
}

func sendIntervalSummary(c *targon.Core, h types.Header, uids, scores []types.U16) error {
	stats := make(map[int]*minerStats)
	totalGPUs := 0
	activeNodes := 0

	for uidstr, nodes := range c.VerifiedNodes {
		uid, _ := strconv.Atoi(uidstr)
		if stats[uid] == nil {
			stats[uid] = &minerStats{
				gpuTypes: make(map[string]int),
				uid:      uid,
			}
		}

		for _, node := range nodes {
			if node == nil {
				continue
			}
			activeNodes++
			for _, gpu := range *node.GPUCards {
				gpuLower := strings.ToLower(gpu)
				totalGPUs++
				stats[uid].gpuCount++
				stats[uid].gpuTypes[gpuLower]++
			}
		}
	}

	for i, uid := range uids {
		if val, ok := stats[int(uid)]; ok {
			val.incentive = float64(scores[i]) / float64(setup.U16MAX)
		}
	}

	// Aggregate GPU types across all miners
	gpuTypes := make(map[string]int)
	statsarr := []*minerStats{}
	for _, miner := range stats {
		statsarr = append(statsarr, miner)
		for gpu, count := range miner.gpuTypes {
			gpuTypes[gpu] += count
		}
	}

	sort.Slice(statsarr, func(i, j int) bool {
		return statsarr[i].uid < statsarr[j].uid
	})

	burned := 0.0
	lastuid := uids[len(uids)-1]
	if int(lastuid) == 28 {
		burned = float64(scores[len(scores)-1]) / float64(setup.U16MAX)
	}

	color := "5763719"
	title := fmt.Sprintf("Daily GPU Summary at block %v", h.Number)
	if c.Deps.Env.DEBUG {
		title = fmt.Sprintf("[DEBUG] %s", title)
		color = "15105570"
	}
	desc := fmt.Sprintf(
		"Total Attested GPUs: %d\n"+
			"Active CVM Nodes: %d\n"+
			"Emission Pool: $%.2f\n"+
			"Burned: %.2f%%\n"+
			"GPU Type Breakdown:\n%s\n"+
			"Per Miner Breakdown:\n%s",
		totalGPUs,
		activeNodes,
		*c.EmissionPool,
		burned,
		formatGPUBreakdown(gpuTypes),
		formatMinerBreakdown(statsarr),
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
	return err
}

func formatGPUBreakdown(gpuTypes map[string]int) string {
	var sb strings.Builder
	for gpu, count := range gpuTypes {
		sb.WriteString(fmt.Sprintf("- %s: %d\n", gpu, count))
	}
	return sb.String()
}

func formatMinerBreakdown(stats []*minerStats) string {
	var sb strings.Builder
	for _, miner := range stats {
		if miner.gpuCount == 0 {
			continue
		}
		sb.WriteString(fmt.Sprintf("- Miner %d: %d GPUs, %.2f%%\n", miner.uid, miner.gpuCount, miner.incentive*100))
	}
	return sb.String()
}
