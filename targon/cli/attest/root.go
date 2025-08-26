// Package attest
package attest

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"math/big"
	"net"
	"os"
	"strings"
	"sync"

	"targon/cli/root"
	"targon/cli/shared"
	"targon/internal/cvm"
	"targon/internal/targon"
	"targon/internal/utils"

	"github.com/centrifuge/go-substrate-rpc-client/v4/signature"
	"github.com/centrifuge/go-substrate-rpc-client/v4/types"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"github.com/subtrahend-labs/gobt/client"
	"github.com/subtrahend-labs/gobt/runtime"

	"github.com/docker/docker/api/types/container"
	dockerclient "github.com/docker/docker/client"
	"github.com/docker/go-connections/nat"
)

var (
	uidFlag                  int
	ipFlag                   string
	chainEndpointFlag        string
	chainNetuidFlag          int
	nvidiaAttestEndpointFlag string
)

func init() {
	ipsCmd.Flags().IntVar(&uidFlag, "uid", -1, "Specific uid to grab GPU info for")
	ipsCmd.Flags().StringVar(&ipFlag, "ip", "", "Specific ip address for off chain testing")
	ipsCmd.Flags().StringVar(&chainEndpointFlag, "chain", "wss://entrypoint-finney.opentensor.ai:443", "Set chain endpoint")
	ipsCmd.Flags().IntVar(&chainNetuidFlag, "netuid", 4, "Set chain netuid")
	ipsCmd.Flags().StringVar(&nvidiaAttestEndpointFlag, "nvidia", "http://localhost:3344", "Set nvidia attest endpoint")

	root.RootCmd.AddCommand(ipsCmd)
}

var ipsCmd = &cobra.Command{
	Use:   "attest",
	Short: "Manually attest a miner or ip address",
	Long:  ``,
	Run: func(cmd *cobra.Command, args []string) {
		if uidFlag == -1 && ipFlag == "" {
			_ = cmd.Help()
			return
		}

		config, err := loadConfig()
		if err != nil {
			fmt.Println("Error loading config: " + err.Error())
			os.Exit(1)
		}

		client, err := client.NewClient(chainEndpointFlag)
		if err != nil {
			fmt.Printf("Error creating client: %s\n", err)
			os.Exit(1)
		}

		kp, err := signature.KeyringPairFromSecret(config.ValidatorHotkeyPhrase, 42)
		if err != nil {
			fmt.Println("Error parsing hotkey phrase: " + err.Error())
			os.Exit(1)
		}

		var neuron *runtime.NeuronInfo
		if uidFlag != -1 {
			blockHash, err := client.Api.RPC.Chain.GetBlockHashLatest()
			if err != nil {
				fmt.Println(utils.Wrap("Failed getting blockhash for neurons", err))
				return
			}
			neuron, err = runtime.GetNeuron(client, uint16(chainNetuidFlag), uint16(uidFlag), &blockHash)
			if err != nil {
				fmt.Println(utils.Wrap("Failed getting neurons", err))
				return
			}
		}
		if uidFlag == -1 {
			neuron = &runtime.NeuronInfo{
				UID:    types.NewUCompact(big.NewInt(444)),
				Hotkey: types.AccountID(kp.PublicKey),
			}
			kp, err := signature.KeyringPairFromSecret(config.MinerHotkeyPhrase, 42)
			if err == nil {
				fmt.Println("Using MINER_HOTKEY_PHRASE")
				neuron.Hotkey = types.AccountID(kp.PublicKey)
			} else {
				fmt.Println(utils.Wrap("Failed creating miner keyring par", err))
				fmt.Println("Using HOTKEY_PHRASE")
			}
		}

		contID, err := createNewContainer("ghcr.io/manifold-inc/targon-nvidia-attest:latest")
		if err != nil {
			fmt.Printf("Error creating container: %s\n", err.Error())
			return
		}
		defer func() {
			err = stopContainer(contID)
			if err != nil {
				fmt.Printf("Error stopping container: %s\n", err.Error())
				return
			}
		}()

		attester := cvm.NewAttester(1, kp, "https://tower.targon.com")
		if len(ipFlag) != 0 {

			// Mock Neuron, use self hotkey
			nonce := targon.NewNonce(kp.Address)
			cvmIP := strings.TrimPrefix(ipFlag, "http://")
			cvmIP = strings.TrimSuffix(cvmIP, ":8080")
			attestPayload, err := attester.GetAttestFromNode(utils.AccountIDToSS58(neuron.Hotkey), cvmIP, nonce)
			if err != nil {
				fmt.Println(err.Error())
				return
			}
			gpus, err := attester.VerifyAttestation(attestPayload, nonce, ipFlag)
			if err != nil {
				fmt.Println(utils.Wrap("CVM attest error", err))
				return
			}
			fmt.Printf("node: %s \n", ipFlag)
			fmt.Printf("gpus: %v\n\n", gpus)
			return
		}

		fileInfo, _ := os.Stdin.Stat()
		var nodes []*targon.MinerNode
		if isPipe := (fileInfo.Mode() & os.ModeNamedPipe) != 0; isPipe {
			nodes = GetNodesFromStdin(cmd)
		}
		if len(nodes) == 0 {
			var neuronIPAddr net.IP = neuron.AxonInfo.IP.Bytes()
			n, err := attester.GetNodes(utils.AccountIDToSS58(neuron.Hotkey), fmt.Sprintf("%s:%d", neuronIPAddr.String(), neuron.AxonInfo.Port))
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
				nonce := targon.NewNonce(kp.Address)
				cvmIP := strings.TrimPrefix(n.IP, "http://")
				cvmIP = strings.TrimSuffix(cvmIP, ":8080")
				attestPayload, err := attester.GetAttestFromNode(utils.AccountIDToSS58(neuron.Hotkey), cvmIP, nonce)
				if err != nil {
					fmt.Printf("%s: %s\n", n.IP, err.Error())
					return
				}
				attprint, _ := json.MarshalIndent(attestPayload, "", "  ")
				fmt.Println(string(attprint))
				gpus, err := attester.VerifyAttestation(attestPayload, nonce, ipFlag)
				if err != nil {
					fmt.Printf("%s: %s\n", n.IP, err.Error())
					return
				}
				fmt.Printf("%s: gpus: %v\n", n.IP, gpus)
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
		nodes = append(nodes, &targon.MinerNode{IP: line, Price: 300})
	}
	return nodes
}

type AttestConfig struct {
	ValidatorHotkeyPhrase string
	MinerHotkeyPhrase     string
}

func loadConfig() (*AttestConfig, error) {
	config := &AttestConfig{}

	configStrings := map[string]*string{
		"validator.hotkey_phrase": &config.ValidatorHotkeyPhrase,
		"miner.hotkey_phrase":     &config.MinerHotkeyPhrase,
	}

	for key, value := range configStrings {
		if viper.GetString(key) == "" {
			shared.PromptConfigString(key)
		}
		*value = viper.GetString(key)
	}

	return config, nil
}

func createNewContainer(image string) (string, error) {
	cli, err := dockerclient.NewClientWithOpts(dockerclient.FromEnv, dockerclient.WithAPIVersionNegotiation())
	if err != nil {
		return "", err
	}

	hostBinding := nat.PortBinding{
		HostIP:   "0.0.0.0",
		HostPort: "3344",
	}
	containerPort, err := nat.NewPort("tcp", "80")
	if err != nil {
		panic("Unable to get the port")
	}

	portBinding := nat.PortMap{containerPort: []nat.PortBinding{hostBinding}}

	cont, err := cli.ContainerCreate(
		context.Background(),
		&container.Config{
			Image: image,
			ExposedPorts: map[nat.Port]struct{}{
				containerPort: {},
			},
		},
		&container.HostConfig{
			PortBindings: portBinding,
			AutoRemove:   true,
		}, nil, nil, "nvidia-attest")
	if err != nil {
		panic(err)
	}

	_ = cli.ContainerStart(context.Background(), cont.ID, container.StartOptions{})
	fmt.Printf("Container %s started\n", cont.ID)
	return cont.ID, nil
}

func stopContainer(containerID string) error {
	cli, err := dockerclient.NewClientWithOpts(dockerclient.FromEnv, dockerclient.WithAPIVersionNegotiation())
	if err != nil {
		return err
	}

	err = cli.ContainerRemove(context.Background(), containerID, container.RemoveOptions{Force: true})
	if err != nil {
		return err
	}

	fmt.Printf("Container %s stopped\n", containerID)
	return nil
}
