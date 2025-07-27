package main

import (
	_ "targon/cli/attest"
	_ "targon/cli/get"
	"targon/cli/root"
	_ "targon/cli/config"
	_ "targon/cli/tower"
)

func main() {
	root.Execute()
}
