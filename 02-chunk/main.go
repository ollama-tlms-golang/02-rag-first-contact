package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"math"
	"net/http"
	"net/url"
	"os"
	"sort"

	"github.com/ollama/ollama/api"
)

var (
	FALSE = false
	TRUE  = true
)

func main() {
	ctx := context.Background()

	var ollamaRawUrl string
	if ollamaRawUrl = os.Getenv("OLLAMA_HOST"); ollamaRawUrl == "" {
		ollamaRawUrl = "http://localhost:11434"
	}

	url, _ := url.Parse(ollamaRawUrl)

	client := api.NewClient(url, http.DefaultClient)

	systemInstructions := `You are a role playing games expert like D&D.
	You are the dungeon master of the Chronicles of Aethelgard game.
	If you need information about the Aethelgard and its species, refer only to the provided content.
	`

	question := "Explain the biological compatibility of the Human species"

	context, err := os.ReadFile("../content/chronicles-of-aethelgard.md")
	if err != nil {
		log.Fatalln("😡", err)
	}

	chunks := ChunkText(string(context), 1024, 256)

	vectorStore := []VectorRecord{}
	// Create embeddings from documents and save them in the store
	for idx, chunk := range chunks {
		fmt.Println("📝 Creating embedding nb:", idx)
		fmt.Println("📝 Chunk:", chunk)

		embedding, _ := GetEmbeddingFromChunk(ctx, client, chunk)

		// Save the embedding in the vector store
		record := VectorRecord{
			Prompt:    chunk,
			Embedding: embedding,
		}
		vectorStore = append(vectorStore, record)
	}

	embeddingFromQuestion, _ := GetEmbeddingFromChunk(ctx, client, question)

	// Search similarites between the question and the vectors of the store
	// 1- calculate the cosine similarity between the question and each vector in the store
	similarities := []Similarity{}

	for _, vector := range vectorStore {
		cosineSimilarity, err := CosineSimilarity(embeddingFromQuestion, vector.Embedding)
		if err != nil {
			log.Fatalln("😡", err)
		}

		// append to similarities
		similarities = append(similarities, Similarity{
			Prompt:           vector.Prompt,
			CosineSimilarity: cosineSimilarity,
		})
	}

	// Select the 5 most similar chunks
	// retrieve in similarities the 5 records with the highest cosine similarity
	// sort the similarities
	sort.Slice(similarities, func(i, j int) bool {
		return similarities[i].CosineSimilarity > similarities[j].CosineSimilarity
	})

	// get the first 5 records
	top5Similarities := similarities[:5]

	fmt.Println("🔍 Top 5 similarities:")
	for _, similarity := range top5Similarities {
		fmt.Println("🔍 Prompt:", similarity.Prompt)
		fmt.Println("🔍 Cosine similarity:", similarity.CosineSimilarity)
		fmt.Println("--------------------------------------------------")
	}

	// Create a new context with the top 5 chunks
	newContext := ""
	for _, similarity := range top5Similarities {
		newContext += similarity.Prompt
	}

	// Answer the question with the new context

	// Prompt construction
	messages := []api.Message{
		{Role: "system", Content: systemInstructions},
		{Role: "system", Content: "CONTENT:\n" + newContext},
		{Role: "user", Content: question},
	}

	req := &api.ChatRequest{
		Model:    "qwen2.5:0.5b",
		Messages: messages,
		Options: map[string]interface{}{
			"temperature":    0.0,
			"repeat_last_n":  2,
			"repeat_penalty": 1.8,
			"top_k":          10,
			"top_p":          0.5,
		},
		Stream: &TRUE,
	}

	fmt.Println("🦄 question:", question)
	fmt.Println("🤖 answer:")
	err = client.Chat(ctx, req, func(resp api.ChatResponse) error {
		fmt.Print(resp.Message.Content)
		return nil
	})

	if err != nil {
		log.Fatalln("😡", err)
	}
	fmt.Println()
	fmt.Println()

}

func ChunkText(text string, chunkSize, overlap int) []string {
	chunks := []string{}
	for start := 0; start < len(text); start += chunkSize - overlap {
		end := start + chunkSize
		if end > len(text) {
			end = len(text)
		}
		chunks = append(chunks, text[start:end])
	}
	return chunks
}

type VectorRecord struct {
	Prompt    string    `json:"prompt"`
	Embedding []float64 `json:"embedding"`
}

type Similarity struct {
	Prompt           string
	CosineSimilarity float64
}

func GetEmbeddingFromChunk(ctx context.Context, client *api.Client, doc string) ([]float64, error) {
	embeddingsModel := "snowflake-arctic-embed:33m"

	req := &api.EmbeddingRequest{
		Model:  embeddingsModel,
		Prompt: doc,
	}
	// get embeddings
	resp, err := client.Embeddings(ctx, req)
	if err != nil {
		log.Println("😡:", err)
		return nil, err
	}
	return resp.Embedding, nil
}

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
