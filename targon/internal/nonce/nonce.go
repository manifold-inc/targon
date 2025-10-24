package nonce

import (
	"crypto/sha256"
	"fmt"
	"strings"

	"github.com/google/uuid"
)

func NewNonce(ss58 string) string {
	prefix := sha256.Sum256([]byte(ss58))
	random := strings.ReplaceAll(uuid.NewString(), "-", "")
	return fmt.Sprintf("%x%s", prefix[:16], random)
}
