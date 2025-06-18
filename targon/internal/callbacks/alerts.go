package callbacks

import (
	"fmt"
	"strings"

	"targon/internal/discord"
	"targon/internal/targon"

	"github.com/centrifuge/go-substrate-rpc-client/v4/types"
)

type minerStats struct {
	gpuCount int
	gpuTypes map[string]int
}

func sendDailyGPUSummary(c *targon.Core, h types.Header) {
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
