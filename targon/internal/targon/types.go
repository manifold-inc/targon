package targon

type AttestResponse struct {
	Success   bool                   `json:"success,omitempty"`
	Nonce     string                 `json:"nonce,omitempty"`
	Token     string                 `json:"token,omitempty"`
	Claims    map[string]AttestClaim `json:"claims,omitempty"`
	Validated bool                   `json:"validated,omitempty"`
	GPUs      []AttestGPU            `json:"gpus,omitempty"`
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
