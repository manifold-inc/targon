package main

import (
	_ "targon/cli/attest"
	_ "targon/cli/get"
	"targon/cli/root"
	_ "targon/cli/update"
)

func main() {
	root.Execute()
}
