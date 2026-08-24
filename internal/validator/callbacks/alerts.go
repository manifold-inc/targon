package callbacks

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/manifold-inc/targon/internal/discord"
	"github.com/manifold-inc/targon/internal/validator"
	"github.com/manifold-inc/targon/internal/validator/setup"

	"github.com/centrifuge/go-substrate-rpc-client/v4/types"
)

type minerStats struct {
	gpuCount  int
	gpuTypes  map[string]int
	cpuTypes  map[string]int
	nodeTypes map[string]int
	cpuCount  int
	uid       int
	incentive float64
}

func sendIntervalSummary(c *validator.Core, h types.Header, uids, scores []uint16) error {
	stats := make(map[int]*minerStats)
	totalGPUs := 0
	totalCPUs := 0
	totalNodes := 0

	for uidstr, nodes := range c.VerifiedNodes {
		uid, _ := strconv.Atoi(uidstr)
		if stats[uid] == nil {
			stats[uid] = &minerStats{
				gpuTypes:  make(map[string]int),
				cpuTypes:  make(map[string]int),
				nodeTypes: make(map[string]int),
				uid:       uid,
			}
		}

		for _, node := range nodes {
			if node == nil {
				continue
			}
			totalNodes++
			nodeType := strings.ToLower(strings.TrimSpace(node.NodeType))
			if nodeType == "" {
				nodeType = "unknown"
			}
			stats[uid].nodeTypes[nodeType]++
			for _, gpu := range *node.GPUCards {
				gpuLower := strings.ToLower(gpu)
				totalGPUs++
				stats[uid].gpuCount++
				stats[uid].gpuTypes[gpuLower]++
			}
			// Only count CPUs for CPU-only nodes;
			if len(*node.GPUCards) == 0 {
				for _, cpu := range *node.CPUCards {
					cpuLower := strings.ToLower(cpu)
					totalCPUs++
					stats[uid].cpuCount++
					stats[uid].cpuTypes[cpuLower]++
				}
			}
		}
	}

	for i, uid := range uids {
		if val, ok := stats[int(uid)]; ok {
			val.incentive = float64(scores[i]) / float64(setup.U16MAX)
		}
	}

	// Aggregate type -> uid -> count across all miners.
	gpuTypeMiners := make(map[string]map[int]int)
	cpuTypeMiners := make(map[string]map[int]int)
	nodeTypeMiners := make(map[string]map[int]int)
	statsarr := []*minerStats{}
	for _, miner := range stats {
		statsarr = append(statsarr, miner)
		accumulate(gpuTypeMiners, miner.uid, miner.gpuTypes)
		accumulate(cpuTypeMiners, miner.uid, miner.cpuTypes)
		accumulate(nodeTypeMiners, miner.uid, miner.nodeTypes)
	}

	sort.Slice(statsarr, func(i, j int) bool {
		return statsarr[i].uid < statsarr[j].uid
	})

	burned := 0.0
	for i, uid := range uids {
		if _, ok := c.BurnDistribution[int(uid)]; ok {
			burned += float64(scores[i]) / float64(setup.U16MAX)
		}
	}
	burnedLine := fmt.Sprintf("%.2f%%", burned*100)
	minerPayout := 0.0
	for _, bids := range c.AuctionResults {
		for _, bid := range bids {
			minerPayout += bid.Payout * 1.2
		}
	}

	color := "5763719"
	title := fmt.Sprintf("Daily Summary at block %v", h.Number)
	if c.Deps.Env.Debug {
		title = fmt.Sprintf("[DEBUG] %s", title)
		color = "15105570"
	}
	desc := fmt.Sprintf(
		"Total Attested GPUs: %d\n"+
			"Total Attested CPUs: %d\n"+
			"Total Nodes: %d\n"+
			"Emission Pool: $%.2f\n"+
			"Miner Payout: $%.2f\n"+
			"Burned: %s\n"+
			"\n"+
			"Node Type Breakdown:\n%s\n"+
			"GPU Type Breakdown:\n%s\n"+
			"CPU Type Breakdown:\n%s\n"+
			"Per Miner Breakdown:\n%s",
		totalGPUs,
		totalCPUs,
		totalNodes,
		*c.EmissionPool,
		minerPayout,
		burnedLine,
		formatTypeBreakdown(nodeTypeMiners),
		formatTypeBreakdown(gpuTypeMiners),
		formatTypeBreakdown(cpuTypeMiners),
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
	err := discord.SendDiscordMessage(c.Deps.Env.DiscordURL, msg)
	c.Deps.Log.Infow("Sent daily GPU summary",
		"total_gpus", totalGPUs,
		"total_cpus", totalCPUs,
		"total_nodes", totalNodes,
		"gpu_types", gpuTypeMiners,
		"cpu_types", cpuTypeMiners,
		"node_types", nodeTypeMiners,
	)
	return err
}

func accumulate(dst map[string]map[int]int, uid int, src map[string]int) {
	for name, count := range src {
		if dst[name] == nil {
			dst[name] = make(map[int]int)
		}
		dst[name][uid] += count
	}
}

// formatTypeBreakdown renders entries of the form:
// - name · total → M{uid}={count} • M{uid}={count}
// sorted by total desc (name asc as tiebreaker), with miners sorted by uid asc.
func formatTypeBreakdown(typeMiners map[string]map[int]int) string {
	if len(typeMiners) == 0 {
		return ""
	}
	type entry struct {
		name   string
		total  int
		miners map[int]int
	}
	entries := make([]entry, 0, len(typeMiners))
	for name, miners := range typeMiners {
		total := 0
		for _, c := range miners {
			total += c
		}
		entries = append(entries, entry{name: name, total: total, miners: miners})
	}
	sort.Slice(entries, func(i, j int) bool {
		if entries[i].total != entries[j].total {
			return entries[i].total > entries[j].total
		}
		return entries[i].name < entries[j].name
	})

	var sb strings.Builder
	for _, e := range entries {
		uids := make([]int, 0, len(e.miners))
		for uid := range e.miners {
			uids = append(uids, uid)
		}
		sort.Ints(uids)

		parts := make([]string, 0, len(uids))
		for _, uid := range uids {
			parts = append(parts, fmt.Sprintf("M%d=%d", uid, e.miners[uid]))
		}
		sb.WriteString(fmt.Sprintf("- %s · %d → %s\n", e.name, e.total, strings.Join(parts, " • ")))
	}
	return sb.String()
}

func formatMinerBreakdown(stats []*minerStats) string {
	var sb strings.Builder
	for _, miner := range stats {
		if miner.gpuCount == 0 && miner.cpuCount == 0 {
			continue
		}
		parts := []string{}
		if miner.gpuCount > 0 {
			parts = append(parts, fmt.Sprintf("%d GPUs", miner.gpuCount))
		}
		if miner.cpuCount > 0 {
			parts = append(parts, fmt.Sprintf("%d CPUs", miner.cpuCount))
		}
		parts = append(parts, fmt.Sprintf("%.2f%%", miner.incentive*100))
		sb.WriteString(fmt.Sprintf("- Miner-%d → %s\n", miner.uid, strings.Join(parts, " • ")))
	}
	return sb.String()
}
