I'll help you create a function to calculate the cosine similarity between two embeddings in Go.



```go
package similarity

import (
	"errors"
	"math"
)

// CosineSimilarity calculates the cosine similarity between two vectors
// Returns a value between -1 and 1, where:
// 1 means vectors are identical
// 0 means vectors are perpendicular
// -1 means vectors are opposite
func CosineSimilarity(vec1, vec2 []float64) (float64, error) {
	if len(vec1) != len(vec2) {
		return 0, errors.New("vectors must have the same length")
	}
	
	// Calculate dot product
	dotProduct := 0.0
	magnitude1 := 0.0
	magnitude2 := 0.0
	
	for i := 0; i < len(vec1); i++ {
		dotProduct += vec1[i] * vec2[i]
		magnitude1 += vec1[i] * vec1[i]
		magnitude2 += vec2[i] * vec2[i]
	}
	
	magnitude1 = math.Sqrt(magnitude1)
	magnitude2 = math.Sqrt(magnitude2)
	
	// Check for zero magnitudes to avoid division by zero
	if magnitude1 == 0 || magnitude2 == 0 {
		return 0, errors.New("vector magnitude cannot be zero")
	}
	
	return dotProduct / (magnitude1 * magnitude2), nil
}

// CosineDistance calculates the cosine distance between two vectors
// Returns a value between 0 and 2, where:
// 0 means vectors are identical
// 1 means vectors are perpendicular
// 2 means vectors are opposite
func CosineDistance(vec1, vec2 []float64) (float64, error) {
	similarity, err := CosineSimilarity(vec1, vec2)
	if err != nil {
		return 0, err
	}
	return 1 - similarity, nil
}

// Example usage:
func Example() {
	embedding1 := []float64{0.010071029, -0.0017594862, 0.05007221}
	embedding2 := []float64{0.009071029, -0.0027594862, 0.04907221}
	
	distance, err := CosineDistance(embedding1, embedding2)
	if err != nil {
		// Handle error
	}
	// Use distance value
	_ = distance
}

```

I've created two functions:

1. `CosineSimilarity`: Returns a value between -1 and 1
   - 1: vectors are identical
   - 0: vectors are perpendicular
   - -1: vectors are opposite

2. `CosineDistance`: Returns a value between 0 and 2
   - 0: vectors are identical
   - 1: vectors are perpendicular
   - 2: vectors are opposite

The distance is calculated as `1 - similarity`. For RAG applications, you'll typically want to use `CosineDistance`, where smaller values indicate more similar embeddings.

You can use it like this:

```go
distance, err := CosineDistance(embedding1, embedding2)
if err != nil {
    log.Fatal(err)
}
fmt.Printf("Distance: %f\n", distance)
```

The implementation includes error handling for:
- Vectors of different lengths
- Zero magnitude vectors

Would you like me to add any additional features or optimizations to the implementation?