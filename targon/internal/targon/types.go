package targon

type AttestPayload struct {
	Attest *AttestResponse
}

type MinerInfo struct {
	Core         *Core    `bson:"inline"`
	Block        int      `bson:"block"`
	Scores       []uint16 `bson:"scores,omitempty"`
	EmissionPool float64  `bson:"emission_pool,omitempty"`
	TaoPrice     float64  `bson:"tao_price,omitempty"`
	Timestamp    int64    `bson:"timestamp,omitempty"`
	Weights      Weights  `bson:"weights,omitempty"`
}
type Weights struct {
	UIDs       []uint16  `bson:"uids"`
	Incentives []float64 `bson:"incentives"`
}

type AttestResponse struct {
	GPULocal struct {
		AttestationResult bool   `json:"attestation_result"`
		Token             string `json:"token"`
		Valid             bool   `json:"valid"`
	} `json:"gpu_local"`
	SwitchLocal struct {
		AttestationResult bool   `json:"attestation_result"`
		Token             string `json:"token"`
		Valid             bool   `json:"valid"`
	} `json:"switch_local"`
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

type AttestClaim struct {
	HWModel            string `json:"hw_model,omitempty"`
	DriverVersion      string `json:"driver_version,omitempty"`
	VBiosVersion       string `json:"vbios_version,omitempty"`
	Measres            string `json:"measres,omitempty"`
	AttestationSuccess bool   `json:"attestation_success,omitempty"`
}

type GPUAttestationResponse struct {
	GPUIds                   []string `json:"gpu_ids"`
	GPUAttestationSuccess    bool     `json:"gpu_attestation_success"`
	SwitchAttestationSuccess bool     `json:"switch_attestation_success"`
	GPUClaims                map[string]struct {
		GPUType string `json:"gpu_type"`
	} `json:"gpu_claims,omitempty"`
	SwitchClaims map[string]struct {
		SwitchType string `json:"switch_type"`
		SwitchID   string `json:"switch_id"`
	} `json:"switch_claims,omitempty"`
}
