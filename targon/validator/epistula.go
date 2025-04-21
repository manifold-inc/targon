package validator

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"time"

	"github.com/centrifuge/go-substrate-rpc-client/v4/signature"
	"github.com/centrifuge/go-substrate-rpc-client/v4/types"
	"github.com/google/uuid"
)

func sha256Hash(str []byte) string {
	h := sha256.New()
	h.Write(str)
	sum := h.Sum(nil)
	return hex.EncodeToString(sum)
}

// Takes sender ss58, sender public, sender private, receiver ss58 and body
func GetEpistulaHeaders(kp signature.KeyringPair, rSS58 string, body []byte) (map[string]string, error) {
	timestamp := time.Now().UnixMilli()
	uuid := uuid.New().String()
	timestampInterval := int64(math.Ceil(float64(timestamp) / 1e4))
	bodyHash := sha256Hash(body)
	message := fmt.Sprintf("%s.%s.%d.%s", bodyHash, uuid, timestamp, rSS58)
	requestSignature, err := signature.Sign([]byte(message), kp.URI)
	if err != nil {
		return nil, err
	}

	s1, _ := signature.Sign(fmt.Appendf([]byte{}, "%d.%s", timestampInterval-1, kp.Address), kp.URI)
	s2, _ := signature.Sign(fmt.Appendf([]byte{}, "%d.%s", timestampInterval, kp.Address), kp.URI)
	s3, _ := signature.Sign(fmt.Appendf([]byte{}, "%d.%s", timestampInterval+1, kp.Address), kp.URI)

	headers := map[string]string{
		"Epistula-Version":            "2",
		"Epistula-Timestamp":          fmt.Sprintf("%d", timestamp),
		"Epistula-Uuid":               uuid,
		"Epistula-Signed-By":          kp.Address,
		"Epistula-Signed-For":         rSS58,
		"Epistula-Request-Signature":  types.NewSignature(requestSignature).Hex(),
		"Epistula-Secret-Signature-0": types.NewSignature(s1).Hex(),
		"Epistula-Secret-Signature-1": types.NewSignature(s2).Hex(),
		"Epistula-Secret-Signature-2": types.NewSignature(s3).Hex(),
		"Content-Type":                "application/json",
		"Connection":                  "keep-alive",
	}

	return headers, nil
}
