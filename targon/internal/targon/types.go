package targon

type MinerBid struct {
	Ip       string   `bson:"ip"`
	Price    int      `bson:"price"`
	UID      string   `bson:"uid"`
	Payout   float64  `bson:"payout"`
	Diluted  bool     `bson:"diluted"`
	UserData UserData `bson:"user_data"`
}

type MinerInfo struct {
	Core      *Core    `bson:"inline"`
	Block     int      `bson:"block"`
	Scores    []uint16 `bson:"scores,omitempty"`
	Timestamp int64    `bson:"timestamp,omitempty"`
	Weights   Weights  `bson:"weights,omitempty"`
}

type Weights struct {
	UIDs       []uint16  `bson:"uids"`
	Incentives []float64 `bson:"incentives"`
}

type UserData struct {
	// Added in attester
	GPUCards     *Cards        `json:"gpu_cards,omitempty"`
	CPUCards     *Cards        `json:"cpu_cards,omitempty"`
	NodeType     string        `json:"node_type"`
	NVCCResponse *NVCCResponse `json:"attestation,omitempty"`

	// Added in handler
	Nonce string `json:"nonce"`
	CVMID string `json:"cvm_id"`
}

type NVCCResponse struct {
	GPURemote struct {
		AttestationResult bool   `json:"attestation_result"`
		Token             string `json:"token"`
		Valid             bool   `json:"valid"`
	} `json:"gpu_remote"`
	SwitchRemote struct {
		AttestationResult bool   `json:"attestation_result"`
		Token             string `json:"token"`
		Valid             bool   `json:"valid"`
	} `json:"switch_remote"`
}

type Cards []string
