package main

import (
	"fmt"
	"strings"

	"github.com/google/uuid"
)

func main() {
	u := uuid.New()
	hexUUIDNoHyphens := strings.ReplaceAll(u.String(), "-", "")
	fmt.Println(len(hexUUIDNoHyphens))
	fmt.Println(hexUUIDNoHyphens)
}
