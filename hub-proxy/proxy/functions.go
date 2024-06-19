package main

import (
	"bufio"
	"bytes"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ChainSafe/go-schnorrkel"
	"github.com/google/uuid"
	"github.com/nitishm/go-rejson/v4"
	"golang.org/x/crypto/sha3"
)

func safeEnv(env string) string {
	// Lookup env variable, and panic if not present

	res, present := os.LookupEnv(env)
	if !present {
		log.Fatalf("Missing environment variable %s", env)
	}
	return res
}

func signMessage(message string, public string, private string) string {
	// Signs a message via schnorrkel pub and private keys

	var pubk [32]byte
	data, err := hex.DecodeString(public)
	if err != nil {
		log.Fatalf("Failed to decode public key: %s", err)
	}
	copy(pubk[:], data)

	var prik [32]byte
	data, err = hex.DecodeString(private)
	if err != nil {
		log.Fatalf("Failed to decode private key: %s", err)
	}
	copy(prik[:], data)

	priv := schnorrkel.SecretKey{}
	priv.Decode(prik)
	pub := schnorrkel.PublicKey{}
	pub.Decode(pubk)

	signingCtx := []byte("substrate")
	signingTranscript := schnorrkel.NewSigningContext(signingCtx, []byte(message))
	sig, _ := priv.Sign(signingTranscript)
	sigEncode := sig.Encode()
	out := hex.EncodeToString(sigEncode[:])

	return "0x" + out
}

func sha256Hash(str string) string {
	// hash a string via sha256

	h := sha3.New256()
	h.Write([]byte(str))
	sum := h.Sum(nil)
	return hex.EncodeToString(sum)
}

func formatListToPythonString(list []string) string {
	// Take a go list of strings and convert it to a pythonic version of the
	// string representaton of a list.

	strList := "["
	for i, element := range list {
		element = strconv.Quote(element)
		element = strings.TrimPrefix(element, "\"")
		element = strings.TrimSuffix(element, "\"")
		separator := "'"
		if strings.ContainsRune(element, '\'') && !strings.ContainsRune(element, '"') {
			separator = "\""
		} else {
			element = strings.ReplaceAll(element, "'", "\\'")
			element = strings.ReplaceAll(element, "\\\"", "\"")
		}
		if i != 0 {
			strList += ", "
		}
		strList += separator + element + separator
	}
	strList += "]"
	return strList
}

func sendEvent(c *Context, data map[string]any) {
	// Send SSE event to response

	eventId := uuid.New().String()
	fmt.Fprintf(c.Response(), "id: %s\n", eventId)
	fmt.Fprintf(c.Response(), "event: new_message\n")
	eventData, _ := json.Marshal(data)
	fmt.Fprintf(c.Response(), "data: %s\n", string(eventData))
	fmt.Fprintf(c.Response(), "retry: %d\n\n", 1500)
	c.Response().Flush()
}

func buildPrompt(messages []RequestBodyMessages) string {
	// Convert openAI api format to simple query string.
	// Temporary untill targon v2

	prompt := ""
	for _, message := range messages {
		prompt += fmt.Sprintf("%s: %s\n", message.Role, message.Content)
	}
	return prompt
}

func getTopMiners(c *Context) []Miner {
	rh := rejson.NewReJSONHandler()
	rh.SetGoRedisClientWithContext(c.Request().Context(), client)
	minerJSON, err := rh.JSONGet("miners", ".")
	if err != nil {
		c.Err.Printf("Failed to JSONGet: %s\n", err.Error())
		return nil
	}

	var miners []Miner
	err = json.Unmarshal(minerJSON.([]byte), &miners)
	if err != nil {
		c.Err.Printf("Failed to JSON Unmarshal: %s\n", err.Error())
		return nil
	}
	return miners
}

func queryMiners(c *Context, req RequestBody) int {
	// Query miners with llm request

	// First we get our miners
	miners := getTopMiners(c)
	if miners == nil {
		return 500
	}

	// For now, we have dummy sources. this will be taken out later
	sources := []string{"https://google.com"}

	// Build the rest of the body hash
	formattedSourcesList := formatListToPythonString(sources)
	prompt := buildPrompt(req.Messages)
	var hashes []string
	hashes = append(hashes, sha256Hash(formattedSourcesList))
	hashes = append(hashes, sha256Hash(prompt))
	bodyHash := sha256Hash(strings.Join(hashes, ""))

	// We use a channel to process requests as they come in by oder of speed
	response := make(chan MinerResponse)

	// Waitgroup to make sure we wait for all miners to finish or cancel
	var minerWaitGroup sync.WaitGroup
	minerWaitGroup.Add(len(miners))

	// Wait in the background for all miners to finish, and close our channel
	// when they do.
	go func() {
		minerWaitGroup.Wait()
		close(response)
	}()

	tr := &http.Transport{
		MaxIdleConns:      10,
		IdleConnTimeout:   30 * time.Second,
		DisableKeepAlives: false,
	}
	httpClient := http.Client{Transport: tr}

	nonce := time.Now().UnixNano()

	// request context
	ctx := c.Request().Context()

	// just being safe; dont access context from inside a goroutine
	// see https://echo.labstack.com/docs/context#concurrency
	warn := c.Warn

	// query each miner at the same time with the variable context of the
	// parent function via go routines
	for _, m := range miners {
		go func(miner Miner) {
			defer minerWaitGroup.Done()

			// build signed body hash and synapse body
			message := []string{fmt.Sprint(nonce), HOTKEY, miner.Hotkey, INSTANCE_UUID, bodyHash}
			joinedMessage := strings.Join(message, ".")
			signedMessage := signMessage(joinedMessage, PUBLIC_KEY, PRIVATE_KEY)
			port := fmt.Sprint(miner.Port)
			version := 672
			body := InferenceBody{
				Name:           "Inference",
				Timeout:        12.0,
				TotalSize:      0,
				HeaderSize:     0,
				RequiredFields: []string{"sources", "query", "seed"},
				Sources:        sources,
				Query:          prompt,
				BodyHash:       "",
				Dendrite: DendriteOrAxon{
					Ip:            "10.0.0.1",
					Version:       &version,
					Nonce:         &nonce,
					Uuid:          &INSTANCE_UUID,
					Hotkey:        HOTKEY,
					Signature:     &signedMessage,
					Port:          nil,
					StatusCode:    nil,
					StatusMessage: nil,
					ProcessTime:   nil,
				},
				Axon: DendriteOrAxon{
					StatusCode:    nil,
					StatusMessage: nil,
					ProcessTime:   nil,
					Version:       nil,
					Nonce:         nil,
					Uuid:          nil,
					Signature:     nil,
					Ip:            miner.Ip,
					Port:          &port,
					Hotkey:        miner.Hotkey,
				},
				SamplingParams: SamplingParams{
					Seed:                nil,
					Truncate:            nil,
					BestOf:              1,
					DecoderInputDetails: true,
					Details:             false,
					DoSample:            true,
					MaxNewTokens:        req.MaxTokens,
					RepetitionPenalty:   1.0,
					ReturnFullText:      false,
					Stop:                []string{"photographer"},
					Temperature:         .01,
					TopK:                10,
					TopNTokens:          5,
					TopP:                .9999999,
					TypicalP:            .9999999,
					Watermark:           false,
				},
				Completion: nil,
			}

			// Build body json
			endpoint := "http://" + miner.Ip + ":" + fmt.Sprint(miner.Port) + "/Inference"
			out, err := json.Marshal(body)
			r, err := http.NewRequestWithContext(ctx, "POST", endpoint, bytes.NewBuffer(out))
			if err != nil {
				warn.Printf("Failed miner request building: %s\n", err.Error())
				return
			}

			// Add all our headers
			// see https://pkg.go.dev/net/http#Header.Add for difference between
			// setting and adding. TLDR case casting; go is a bit opinionated
			r.Close = true
			r.Header["Content-Type"] = []string{"application/json"}
			r.Header["Connection"] = []string{"keep-alive"}
			r.Header["name"] = []string{"Inference"}
			r.Header["timeout"] = []string{"12.0"}
			r.Header["bt_header_axon_ip"] = []string{miner.Ip}
			r.Header["bt_header_axon_port"] = []string{strconv.Itoa(miner.Port)}
			r.Header["bt_header_axon_hotkey"] = []string{miner.Hotkey}
			r.Header["bt_header_dendrite_ip"] = []string{"0.0.0.0"}
			r.Header["bt_header_dendrite_version"] = []string{"672"}
			r.Header["bt_header_dendrite_nonce"] = []string{strconv.Itoa(int(nonce))}
			r.Header["bt_header_dendrite_uuid"] = []string{INSTANCE_UUID}
			r.Header["bt_header_dendrite_hotkey"] = []string{HOTKEY}
			r.Header["bt_header_input_obj_sources"] = []string{"W10="}
			r.Header["bt_header_input_obj_query"] = []string{"IiI="}
			r.Header["bt_header_dendrite_signature"] = []string{signedMessage}
			r.Header["header_size"] = []string{"0"}
			r.Header["total_size"] = []string{"0"}
			r.Header["computed_body_hash"] = []string{bodyHash}
			r.Header.Add("Accept-Encoding", "identity")

			// Send request
			res, err := httpClient.Do(r)

			// Handle error sending request in general
			if err != nil {
				warn.Printf("Miner: %s %s\nError: %s\n", miner.Hotkey, miner.Coldkey, err.Error())
				if res != nil {
					res.Body.Close()
				}
				return
			}

			// Handle non 200 response code
			if res.StatusCode != http.StatusOK {
				bdy, _ := io.ReadAll(res.Body)
				res.Body.Close()
				warn.Printf("Miner: %s %s\nError: %s\n", miner.Hotkey, miner.Coldkey, string(bdy))
				return
			}

			// For some reason, axon's with a version below this fail requests, atleast
			// at time of building
			axon_version := res.Header.Get("Bt_header_axon_version")
			ver, err := strconv.Atoi(axon_version)
			if err != nil || ver < 672 {
				res.Body.Close()
				warn.Printf("Miner: %s %s\nError: Axon version too low\n", miner.Hotkey, miner.Coldkey)
				return
			}

			// Successfull request; send to channel for processing
			response <- MinerResponse{Res: res, ColdKey: miner.Coldkey, HotKey: miner.Hotkey}
		}(m)
	}

	attempts := 0
	for {
		attempts++

		// this blocks untill we get a response from a miner, or the channel closes
		res, more := <-response
		if !more {
			c.Warn.Printf("All miners failed query")
			return 500
		}
		c.Info.Printf("Attempt: %d Miner: %s %s\n", attempts, res.HotKey, res.ColdKey)

		// stream body in via reader and parse / send tokens
		reader := bufio.NewReader(res.Res.Body)
		finished := false
		for {
			token, err := reader.ReadString(' ')
			if strings.Contains(token, "<s>") || strings.Contains(token, "</s>") || strings.Contains(token, "<im_end>") {
				finished = true
				token = strings.ReplaceAll(token, "<s>", "")
				token = strings.ReplaceAll(token, "</s>", "")
				token = strings.ReplaceAll(token, "<im_end>", "")
			}
			if err != nil && err != io.EOF {
				c.Err.Println(err.Error())
				break
			}
			sendEvent(c, map[string]any{
				"type":     "answer",
				"text":     token,
				"finished": finished,
			})
			if err == io.EOF {
				break
			}
		}
		res.Res.Body.Close()

		// If we arent finished, this probably means something failed. We try again
		// streaming from the next miner. This is a bit broken in the case that
		// a miner fails 1/2 way through a stream, and the second miner starts a new
		// stream from the beginning. From testing, this rarely happens, and failures
		// are usually due to no outputs from miners in the first place
		if finished == false {
			continue
		}

		// if we finished, break as we dont need to continue reading responses
		break
	}
	for {
		// Catch all remaining responses. This would be better if we made a context
		// and canceled the remaining ones. Oh well.
		select {
		case res, ok := <-response:
			if !ok {
				response = nil
				break
			}
			res.Res.Body.Close()
		}
		if response == nil {
			break
		}
	}
	return 200
}
