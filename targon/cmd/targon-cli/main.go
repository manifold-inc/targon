package main

import (
	_ "targon/cli/attest"
	_ "targon/cli/config"
	_ "targon/cli/get"
	_ "targon/cli/vali"
	"targon/cli/root"
)

func main() {
	root.Execute()
}
