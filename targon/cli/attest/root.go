package cli

import (
	"bufio"
	"fmt"
	"math/big"
	"net"
	"os"
	"strings"
	"sync"

	"targon/cli/root"
	"targon/internal/cvm"
	"targon/internal/setup"
	sutils "targon/internal/subtensor/utils"
	"targon/internal/targon"
	"targon/internal/utils"

	"github.com/centrifuge/go-substrate-rpc-client/v4/signature"
	"github.com/centrifuge/go-substrate-rpc-client/v4/types"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"github.com/subtrahend-labs/gobt/runtime"
	"go.uber.org/zap"
)

var (
	uidflag int
	ipflag  string
)

func init() {
	ipsCmd.Flags().IntVar(&uidflag, "uid", -1, "Specific uid to grab GPU info for")
	ipsCmd.Flags().StringVar(&ipflag, "ip", "", "Specific ip address for off chain testing")

	root.RootCmd.AddCommand(ipsCmd)
}

var ipsCmd = &cobra.Command{
	Use:   "attest",
	Short: "Manually attest a miner or ip address",
	Long:  ``,
	Run: func(cmd *cobra.Command, args []string) {
		deps := setup.Init(zap.FatalLevel)
		core := targon.CreateCore(deps)

		if uidflag == -1 && ipflag == "" {
			fmt.Println("Please specify uid or ip")
			return
		}

		var neuron *runtime.NeuronInfo
		if uidflag != -1 {
			blockHash, err := core.Deps.Client.Api.RPC.Chain.GetBlockHashLatest()
			if err != nil {
				fmt.Println(utils.Wrap("Failed getting blockhash for neurons", err))
				return
			}
			NETUID := viper.GetInt("netuid")
			neuron, err = runtime.GetNeuron(core.Deps.Client, uint16(NETUID), uint16(uidflag), &blockHash)
			if err != nil {
				fmt.Println(utils.Wrap("Failed getting neurons", err))
				return
			}
		}
		if uidflag == -1 {
			neuron = &runtime.NeuronInfo{
				UID:    types.NewUCompact(big.NewInt(444)),
				Hotkey: types.AccountID(deps.Hotkey.PublicKey),
			}
			HOTKEY_PHRASE := viper.GetString("hotkey_phrase")
			kp, err := signature.KeyringPairFromSecret(HOTKEY_PHRASE, 42)
			if err == nil {
				fmt.Println("Using MINER_HOTKEY_PHRASE")
				neuron.Hotkey = types.AccountID(kp.PublicKey)
			} else {
				fmt.Println(utils.Wrap("Failed creating miner keyring par", err))
				fmt.Println("Using HOTKEY_PHRASE")
			}
		}

		NVIDIA_ATTEST_ENDPOINT := viper.GetString("nvidia_attest_endpoint")
		attester := cvm.NewAttester(1, core.Deps.Hotkey, NVIDIA_ATTEST_ENDPOINT)
		if len(ipflag) != 0 {

			// Mock Neuron, use self hotkey
			nonce := targon.NewNonce(core.Deps.Hotkey.Address)
			cvmIP := strings.TrimPrefix(ipflag, "http://")
			cvmIP = strings.TrimSuffix(cvmIP, ":8080")
			attestPayload, err := attester.GetAttestFromNode(sutils.AccountIDToSS58(neuron.Hotkey), cvmIP, nonce)
			if err != nil {
				fmt.Println(err.Error())
				return
			}
			gpus, _, err := attester.CheckAttest(attestPayload, nonce)
			if err != nil {
				fmt.Println(utils.Wrap("CVM attest error", err))
				return
			}
			fmt.Printf("node: %s \n", ipflag)
			fmt.Printf("gpus: %v\n\n", gpus)
			return
		}
		fileInfo, _ := os.Stdin.Stat()
		var nodes []*targon.MinerNode
		if isPipe := (fileInfo.Mode() & os.ModeNamedPipe) != 0; isPipe {
			nodes = GetNodesFromStdin(cmd)
		}
		if len(nodes) == 0 {
			var neuronIpAddr net.IP = neuron.AxonInfo.IP.Bytes()
			n, err := attester.GetNodes(sutils.AccountIDToSS58(neuron.Hotkey), fmt.Sprintf("%s:%d", neuronIpAddr.String(), neuron.AxonInfo.Port))
			if err != nil {
				panic(err)
			}
			nodes = n
		}
		fmt.Printf("Nodes: %v\n", nodes)
		fmt.Println("CVM attest results")
		wg := sync.WaitGroup{}
		wg.Add(len(nodes))

		for _, n := range nodes {
			go func() {
				defer wg.Done()
				nonce := targon.NewNonce(core.Deps.Hotkey.Address)
				cvmIP := strings.TrimPrefix(n.Ip, "http://")
				cvmIP = strings.TrimSuffix(cvmIP, ":8080")
				attestPayload, err := attester.GetAttestFromNode(sutils.AccountIDToSS58(neuron.Hotkey), cvmIP, nonce)
				if err != nil {
					fmt.Printf("%s: %s\n", n.Ip, err.Error())
					return
				}
				gpus, _, err := attester.CheckAttest(attestPayload, nonce)
				if err != nil {
					fmt.Printf("%s: %s\n", n.Ip, err.Error())
					return
				}
				fmt.Printf("%s: gpus: %v\n", n.Ip, gpus)
			}()
		}
		wg.Wait()
	},
}

func GetNodesFromStdin(cmd *cobra.Command) []*targon.MinerNode {
	inputReader := cmd.InOrStdin()
	scanner := bufio.NewScanner(inputReader)
	nodes := []*targon.MinerNode{}
	for scanner.Scan() {
		line := scanner.Text()
		nodes = append(nodes, &targon.MinerNode{Ip: line, Price: 300})
	}
	return nodes
}
