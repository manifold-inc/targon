package targon

type AttestResponse struct {
	AttestationResult bool   `json:"attestation_result,omitempty"`
	Token             string `json:"token,omitempty"`
	Valid             bool   `json:"valid,omitempty"`
}

type AttestClaim struct {
	HWModel            string `json:"hw_model,omitempty"`
	DriverVersion      string `json:"driver_version,omitempty"`
	VBiosVersion       string `json:"vbios_version,omitempty"`
	Measres            string `json:"measres,omitempty"`
	AttestationSuccess bool   `json:"attestation_success,omitempty"`
}

type AttestGPU struct {
	Id     string        `json:"id,omitempty"`
	Model  string        `json:"model,omitempty"`
	Claims AttestGPUInfo `json:"claims,omitempty"`
}

type AttestGPUInfo struct {
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
	} `json:"gpu_claims,omitempty"`
}
