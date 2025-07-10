package targon

import "github.com/subtrahend-labs/gobt/boilerplate"

func SetMainFunc(v *boilerplate.BaseChainSubscriber, c *Core) {
	v.SetMainFunc(func(i <-chan bool, o chan<- bool) {
		mainFunc(c, i, o)
	})
}

// This always runs sync with log callbacks, so it is thread safe
func mainFunc(c *Core, i <-chan bool, o chan<- bool) {
	<-i
	c.Deps.Log.Info("Shuting down...")
	err := SaveMongoBackup(c)
	if err != nil {
		c.Deps.Log.Errorw("Failed saving backup of state", "error", err)
	}
	o <- true
}
