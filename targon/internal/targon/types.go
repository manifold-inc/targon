package targon

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
	GPUAttestationSuccess    bool `json:"gpu_attestation_success"`
	SwitchAttestationSuccess bool `json:"switch_attestation_success"`
	GPUClaims                map[string]struct {
		GPUType string `json:"gpu_type"`
		GPUID   string `json:"gpu_id"`
	} `json:"gpu_claims,omitempty"`
	SwitchClaims map[string]struct {
		SwitchType string `json:"switch_type"`
		SwitchID   string `json:"switch_id"`
	} `json:"switch_claims,omitempty"`
}
