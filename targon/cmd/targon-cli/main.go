package main

import (
	_ "targon/cli/attest"
	_ "targon/cli/config"
	_ "targon/cli/get"
	"targon/cli/root"
)

func main() {
	root.Execute()
}
