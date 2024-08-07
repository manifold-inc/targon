package main

import (
	"bufio"
	"bytes"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
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

func sendEvent(c *Context, data string) {
	// Send SSE event to response
	fmt.Fprintf(c.Response(), "data: %s\n\n", data)
	c.Response().Flush()
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

func queryMiners(c *Context, req RequestBody) (ResponseInfo, error) {
	// Query miners with llm request

	// First we get our miners
	miners := getTopMiners(c)
	if miners == nil {
		return ResponseInfo{}, errors.New("No Miners")
	}

	// Build the rest of the body hash
	bodyHash := sha256Hash("")

	tr := &http.Transport{
		MaxIdleConns:      10,
		IdleConnTimeout:   30 * time.Second,
		DisableKeepAlives: false,
	}
	httpClient := http.Client{Transport: tr, Timeout: 10 * time.Second}

	nonce := time.Now().UnixNano()

	messages_json, ok := json.Marshal(req.Messages)
	if ok != nil {
		c.Warn.Printf(ok.Error())
		return ResponseInfo{}, errors.New("Failed to json marshall messages")
	}

	// query each miner at the same time with the variable context of the
	// parent function via go routines
	for index, miner := range miners {
		message := []string{fmt.Sprint(nonce), HOTKEY, miner.Hotkey, INSTANCE_UUID, bodyHash}
		joinedMessage := strings.Join(message, ".")
		signedMessage := signMessage(joinedMessage, PUBLIC_KEY, PRIVATE_KEY)
		version := 710
		body := InferenceBody{
			Name:             "Inference",
			Timeout:          12.0,
			TotalSize:        0,
			HeaderSize:       0,
			RequiredFields:   []string{},
			Messages:         string(messages_json),
			ComputedBodyHash: "",
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
				Port:          &miner.Port,
				Hotkey:        miner.Hotkey,
			},
			SamplingParams: SamplingParams{
				Seed:                5688697,
				Truncate:            nil,
				BestOf:              1,
				DecoderInputDetails: true,
				Details:             false,
				DoSample:            true,
				MaxNewTokens:        req.MaxTokens,
				RepetitionPenalty:   1.0,
				ReturnFullText:      false,
				Stop:                []string{""},
				Temperature:         .01,
				TopK:                10,
				TopNTokens:          5,
				TopP:                .98,
				TypicalP:            .98,
				Watermark:           false,
				Stream:              true,
			},
		}

		endpoint := "http://" + miner.Ip + ":" + fmt.Sprint(miner.Port) + "/Inference"
		out, err := json.Marshal(body)
		r, err := http.NewRequest("POST", endpoint, bytes.NewBuffer(out))
		if err != nil {
			c.Warn.Printf("Failed miner request: %s\n", err.Error())
			continue
		}
		r.Close = true
		r.Header["Content-Type"] = []string{"application/json"}
		r.Header["Connection"] = []string{"keep-alive"}

		r.Header["name"] = []string{"Inference"}
		r.Header["timeout"] = []string{"12.0"}
		r.Header["bt_header_axon_ip"] = []string{miner.Ip}
		r.Header["bt_header_axon_port"] = []string{strconv.Itoa(miner.Port)}
		r.Header["bt_header_axon_hotkey"] = []string{miner.Hotkey}
		r.Header["bt_header_dendrite_ip"] = []string{"10.0.0.1"}
		r.Header["bt_header_dendrite_version"] = []string{fmt.Sprint(version)}
		r.Header["bt_header_dendrite_nonce"] = []string{strconv.Itoa(int(nonce))}
		r.Header["bt_header_dendrite_uuid"] = []string{INSTANCE_UUID}
		r.Header["bt_header_dendrite_hotkey"] = []string{HOTKEY}
		r.Header["bt_header_dendrite_signature"] = []string{signedMessage}
		r.Header["bt_header_input_obj_messages"] = []string{"IiI="}
		r.Header["header_size"] = []string{"0"}
		r.Header["total_size"] = []string{"0"}
		r.Header["computed_body_hash"] = []string{bodyHash}
		r.Header.Add("Accept-Encoding", "identity")

		res, err := httpClient.Do(r)
		if err != nil {
			c.Warn.Printf("Miner: %s %s\nError: %s\n", miner.Hotkey, miner.Coldkey, err.Error())
			if res != nil {
				res.Body.Close()
			}
			continue
		}
		if res.StatusCode != http.StatusOK {
			bdy, _ := io.ReadAll(res.Body)
			res.Body.Close()
			c.Warn.Printf("Miner: %s %s\nError: %s\n", miner.Hotkey, miner.Coldkey, string(bdy))
			continue
		}

		axon_version := res.Header.Get("Bt_header_axon_version")
		ver, err := strconv.Atoi(axon_version)
		if err != nil || ver < 672 {
			res.Body.Close()
			c.Warn.Printf("Miner: %s %s\nError: Axon version too low\n", miner.Hotkey, miner.Coldkey)
			continue
		}

		c.Info.Printf("Attempt: %d Miner: %s %s\n", index, miner.Hotkey, miner.Coldkey)
		reader := bufio.NewReader(res.Body)
		finished := false
		ans := ""
		for {
			token, err := reader.ReadString(' ')
			if strings.Contains(token, "<s>") || strings.Contains(token, "</s>") || strings.Contains(token, "<im_end>") {
				finished = true
				token = strings.ReplaceAll(token, "<s>", "")
				token = strings.ReplaceAll(token, "</s>", "")
				token = strings.ReplaceAll(token, "<im_end>", "")
			}
			ans += token
			if err != nil && err != io.EOF {
				ans = ""
				c.Err.Println(err.Error())
				break
			}

			data := Response{
				Id:      uuid.New().String(),
				Object:  "chat.completion.chunk",
				Created: time.Now().String(),
				Model:   "NousResearch/Meta-Llama-3.1-8B-Instruct",
				Choices: []Choice{{Delta: Delta{Content: token}}},
			}
			eventData, _ := json.Marshal(data)
			sendEvent(c, string(eventData))
			if err == io.EOF {
				sendEvent(c, "[DONE]")
				finished = true
				break
			}
		}
		res.Body.Close()
		if finished == false {
			continue
		}
		return ResponseInfo{Miner: miner, Attempt: index}, nil
	}
	return ResponseInfo{}, errors.New("Ran out of miners to query")
}

func updatOrganicRequest(db *sql.DB,res ResponseInfo, pub_id string) {
	_, err := db.Exec("UPDATE organic_request SET uid=$1, hotkey=$2, coldkey=$3, miner_address=$4, attempt=$5 WHERE pub_id=$6",res.Miner.Uid, res.Miner.Hotkey, res.Miner.Coldkey, fmt.Sprintf("http://%s:%d", res.Miner.Ip, res.Miner.Port), res.Attempt, pub_id)
	if err != nil {
		log.Println("Failed to update")
		log.Println(err)
		return
	}
	return
}
