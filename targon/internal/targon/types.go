package targon

type AttestResponse struct {
	GPU struct {
		AttestationResult bool   `json:"attestation_result"`
		Token             string `json:"token"`
		Valid             bool   `json:"valid"`
	} `json:"gpu"`
	Switch struct {
		AttestationResult bool   `json:"attestation_result"`
		Token             string `json:"token"`
		Valid             bool   `json:"valid"`
	} `json:"switch"`
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
