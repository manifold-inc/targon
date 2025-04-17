package targon

func Normalize(arr []float64, sumTo float64) []float64 {
	sum := 0.0
	for _, num := range arr {
		sum += num
	}
	if sum == 0.0 {
		return arr
	}
	newArr := []float64{}
	for _, num := range arr {
		newArr = append(newArr, (num/sum)*sumTo)
	}
	return newArr
}
