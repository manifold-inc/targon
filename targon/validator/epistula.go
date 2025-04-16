package validator

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"time"

	"github.com/ChainSafe/go-schnorrkel"
	"github.com/google/uuid"
)

func signMessage(message []byte, public string, private string) (string, error) {
	// Signs a message via schnorrkel pub and private keys
	var pubk [32]byte
	data, err := hex.DecodeString(public)
	if err != nil {
		return "", err
	}
	copy(pubk[:], data)

	var prik [32]byte
	data, err = hex.DecodeString(private)
	if err != nil {
		return "", err
	}
	copy(prik[:], data)

	priv := schnorrkel.SecretKey{}
	priv.Decode(prik)
	pub := schnorrkel.PublicKey{}
	pub.Decode(pubk)
	signingCtx := []byte("substrate")
	signingTranscript := schnorrkel.NewSigningContext(signingCtx, message)
	sig, _ := priv.Sign(signingTranscript)
	sigEncode := sig.Encode()
	out := hex.EncodeToString(sigEncode[:])
	return "0x" + out, nil
}

func sha256Hash(str []byte) string {
	h := sha256.New()
	h.Write(str)
	sum := h.Sum(nil)
	return hex.EncodeToString(sum)
}

// Takes sender ss58, sender public, sender private, receiver ss58 and body
func GetEpistulaHeaders(sSS58, sPub, sPriv, rSS58 string, body []byte) (map[string]string, error) {
	timestamp := time.Now().UnixMilli()
	uuid := uuid.New().String()
	timestampInterval := int64(math.Ceil(float64(timestamp) / 1e4))
	bodyHash := sha256Hash(body)
	message := fmt.Sprintf("%s.%s.%d.%s", bodyHash, uuid, timestamp, rSS58)
	requestSignature, err := signMessage([]byte(message), sPub, sPriv)
	if err != nil {
		return nil, err
	}

	s1, _ := signMessage(fmt.Appendf([]byte{}, "%d.%s", timestampInterval-1, sSS58), sPub, sPriv)
	s2, _ := signMessage(fmt.Appendf([]byte{}, "%d.%s", timestampInterval, sSS58), sPub, sPriv)
	s3, _ := signMessage(fmt.Appendf([]byte{}, "%d.%s", timestampInterval+1, sSS58), sPub, sPriv)

	headers := map[string]string{
		"Epistula-Version":            "2",
		"Epistula-Timestamp":          fmt.Sprintf("%d", timestamp),
		"Epistula-Uuid":               uuid,
		"Epistula-Signed-By":          sSS58,
		"Epistula-Signed-For":         rSS58,
		"Epistula-Request-Signature":  requestSignature,
		"Epistula-Secret-Signature-0": s1,
		"Epistula-Secret-Signature-1": s2,
		"Epistula-Secret-Signature-2": s3,
		"Content-Type":                "application/json",
		"Connection":                  "keep-alive",
	}

	return headers, nil
}
