package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"

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

	context, err := os.ReadFile("../content/chronicles-of-aethelgard.md")
	if err != nil {
		log.Fatalln("😡", err)
	}

	//fmt.Println(string(context))

	askMeAnything := func(question string) {
		// Prompt construction
		messages := []api.Message{
			{Role: "system", Content: systemInstructions},
			{Role: "system", Content: "CONTENT:\n" + string(context)},
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

		err := client.Chat(ctx, req, func(resp api.ChatResponse) error {
			fmt.Print(resp.Message.Content)
			return nil
		})

		if err != nil {
			log.Fatalln("😡", err)
		}
		fmt.Println()
		fmt.Println()
	}

	askMeAnything("Explain the biological compatibility of the Human species?")
	fmt.Println("--------------------------------------------------")

}
