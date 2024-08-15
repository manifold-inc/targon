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

type Epistula struct {
	Data      InferenceBody `json:"data"`
	Nonce     int64         `json:"nonce"`
	SignedBy  string        `json:"signed_by"`
	SignedFor string        `json:"signed_for"`
}

type InferenceBody struct {
	Messages       []RequestBodyMessages `json:"messages"`
	SamplingParams SamplingParams        `json:"sampling_params"`
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
