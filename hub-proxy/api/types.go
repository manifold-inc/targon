package main

import "net/http"

type MinerResponse struct {
	Res     *http.Response
	ColdKey string
	HotKey  string
}
type Response struct {
	Id      string   `json:"id"`
	Object  string   `json:"object"`
	Created string   `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
}
type Choice struct {
	Delta Delta `json:"delta"`
}
type Delta struct {
	Content string `json:"content"`
}

type RequestBodyMessages struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	Name    string `json:"name"`
}

type RequestBody struct {
	Model     string                `json:"model"`
	Messages  []RequestBodyMessages `json:"messages"`
	ApiKey    string                `json:"api_key"`
	MaxTokens int                   `json:"max_tokens"`
	PubId     string                `json:"pub_id"`
}

type Miner struct {
	Ip      string `json:"ip,omitempty"`
	Port    int    `json:"port,omitempty"`
	Hotkey  string `json:"hotkey,omitempty"`
	Coldkey string `json:"coldkey,omitempty"`
	Uid     int    `json:"uid,omitempty"`
}

type InferenceBody struct {
	Name             string         `json:"name"`
	Timeout          float32        `json:"timeout"`
	TotalSize        int            `json:"total_size"`
	HeaderSize       int            `json:"header_size"`
	Dendrite         DendriteOrAxon `json:"dendrite"`
	Axon             DendriteOrAxon `json:"axon"`
	ComputedBodyHash string         `json:"computed_body_hash"`
	RequiredFields   []string       `json:"required_hash_fields"`
	Messages         string         `json:"messages"`
	SamplingParams   SamplingParams `json:"sampling_params"`
}

type DendriteOrAxon struct {
	StatusCode    *string `json:"status_code"`
	StatusMessage *string `json:"status_message"`
	ProcessTime   *string `json:"process_time"`
	Ip            string  `json:"ip"`
	Port          *int    `json:"port"`
	Version       *int    `json:"version"`
	Nonce         *int64  `json:"nonce"`
	Uuid          *string `json:"uuid"`
	Hotkey        string  `json:"hotkey"`
	Signature     *string `json:"signature"`
}
type SamplingParams struct {
	BestOf              int      `json:"best_of"`
	DecoderInputDetails bool     `json:"decoder_input_details"`
	Details             bool     `json:"details"`
	DoSample            bool     `json:"do_sample"`
	MaxNewTokens        int      `json:"max_new_tokens"`
	RepetitionPenalty   float32  `json:"repetition_penalty"`
	ReturnFullText      bool     `json:"return_full_text"`
	Stop                []string `json:"stop"`
	Temperature         float32  `json:"temperature"`
	TopK                int      `json:"top_k"`
	TopNTokens          int      `json:"top_n_tokens"`
	TopP                float32  `json:"top_p"`
	TypicalP            float32  `json:"typical_p"`
	Watermark           bool     `json:"watermark"`
	Seed                int      `json:"seed"`
	Truncate            *string  `json:"truncate"`
	Stream              bool     `json:"stream"`
}

type Event struct {
	Event string                 `json:"event"`
	Id    string                 `json:"id"`
	Retry int                    `json:"retry"`
	Data  map[string]interface{} `json:"data"`
}

type ResponseInfo struct {
	Miner   Miner
	Attempt int
}
