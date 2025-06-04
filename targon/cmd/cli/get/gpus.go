package get

import (
	"context"
	"fmt"
	"os"
	"strconv"

	"targon/internal/setup"
	"targon/internal/targon"
	"targon/internal/utils"

	"github.com/spf13/cobra"
	"go.mongodb.org/mongo-driver/v2/bson"
	"go.mongodb.org/mongo-driver/v2/mongo"
	"go.mongodb.org/mongo-driver/v2/mongo/options"
)

var (
	uidflag  int
	allflag  bool
	gpusflag bool
)

func init() {
	ipsCmd.Flags().IntVar(&uidflag, "uid", -1, "Specific uid to grab node ips for")
	ipsCmd.Flags().BoolVar(&allflag, "all", false, "Show all nodes not just ones passing attestation")
	ipsCmd.Flags().BoolVar(&gpusflag, "gpus", false, "Show gpus for that node")
	getCmd.AddCommand(ipsCmd)
}

var ipsCmd = &cobra.Command{
	Use:   "nodes",
	Short: "n",
	Long:  ``,
	Run: func(cmd *cobra.Command, args []string) {
		m, err := setup.InitMongo()
		if err != nil {
			fmt.Println(utils.Wrap("failed connecting to mongo", err))
			os.Exit(-1)
		}
		minerCol := m.Database("targon").Collection("miner_info")
		opts := options.FindOne().SetSort(bson.D{{Key: "block", Value: -1}}) // Sort by 'value' in descending order

		// Find the record with the max value
		var result targon.MinerInfo
		err = minerCol.FindOne(context.TODO(), bson.D{}, opts).Decode(&result)
		if err != nil {
			if err == mongo.ErrNoDocuments {
				fmt.Println("No documents found")
				return
			}
			fmt.Println(err)
			return
		}
		if allflag {
			fmt.Println("not supported yet")
			return
		}

		for uid, nodes := range result.Core.PassedAttestation {
			intuid, _ := strconv.Atoi(uid)
			if uidflag != intuid && uidflag != -1 {
				continue
			}
			if len(nodes) == 0 {
				continue
			}
			fmt.Printf("UID %s nodes\n", uid)
			for node, gpus := range nodes {
				if gpusflag {
					fmt.Printf("%s: %v", node, gpus)
				} else {
					fmt.Println(node)
				}
			}
			fmt.Println()
		}
	},
}
