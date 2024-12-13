# RAG "from Scratch" with Go and Ollama - First Contact

Today we'll discuss RAG (Retrieval Augmented Generation). This is a way to provide knowledge that a LLM (Large Language Model) doesn't have, allowing it to better answer questions about the Bible, your documentation, a very specific domain of knowledge, and so on. And this without having to retrain it (which is costly in time and energy). Some large models have an extensive knowledge base, but they don't know everything, particularly about information created after their training or documentation from your brain that won't be public. And this is even more true for smaller models (these "Tiny models" that I'm so fond of).

I have created, with the help of [Claude.AI](https://claude.ai), content that describes an entire fantastic bestiary oriented towards heroic fantasy, which would be used for a completely fictional role-playing game (Chronicles of Aethelgard). This content is available on [Chronicles of Aethelgard](https://github.com/ollama-tlms-golang/02-rag-first-contact/blob/main/content/chronicles-of-aethelgard.md).

Quick summary of this content (thanks to Claude for the writing 😘):
*The Chronicles of Aethelgard present a fantastic world populated by numerous intelligent species, divided into several categories: noble species (humans, elves, and dwarves), wild species (orcs, halflings, and gnomes), and exotic species (such as drakeid and demons-descended tieflings). Each species has its own biological characteristics, unique culture, and traditions, with varying lifespans ranging from 60-80 years for humans to over 500 years for elves. The relationships between these different species are complex and constantly evolving, alternating between cooperation, distrust, and conflicts, with humans often playing a central intermediary role thanks to their great adaptability and ability to reproduce with several other species.*

And so I'm going to try to feed "Baby Qwen" with this content to see if it can answer questions about this fictional world.

Let's get started!
> Of course, it's better for your understanding if you've read the previous blog posts.

## Let's try in "classic" mode

My first attempt was to load the content into a variable, add it to the prompt for the model, and ask it a question: "Explain the biological compatibility of the Human species". This question involves all parts of the document, as the model must check for compatibility with each described species.

Here's the code in its entirety, and then I'll explain it in detail:

```golang
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

	question := "Explain the biological compatibility of the Human species"

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

	err = client.Chat(ctx, req, func(resp api.ChatResponse) error {
		fmt.Print(resp.Message.Content)
		return nil
	})

	if err != nil {
		log.Fatalln("😡", err)
	}
	fmt.Println()
}
```

### Some explanations

This is a program that interacts with a LLM (`qwen2.5:0.5b`) using Ollama's Go API to create a specialized role-playing assistant.

The section below defines the system instructions for "Baby Qwen" (the LLM):

```go
systemInstructions := `You are a role playing games expert like D&D...`
```

These instructions define the role and context in which the model should operate.

The program then reads a content file containing information about the Aethelgard universe:

```go
context, err := os.ReadFile("../content/chronicles-of-aethelgard.md")
if err != nil {
    log.Fatalln("😡", err)
}
```

Then, the request construction is done by creating an array of messages:
```go
messages := []api.Message{
    {Role: "system", Content: systemInstructions},
    {Role: "system", Content: "CONTENT:\n" + string(context)},
    {Role: "user", Content: question},
}
```

This structure follows the classic format of conversations with language models: system instructions, context, then user question.

The request configuration is particularly interesting:

```go
req := &api.ChatRequest{
    Model:    "qwen2.5:0.5b",    // Uses the Qwen 2.5 model (0.5 billion parameters)
    Messages: messages,
    Options: map[string]interface{}{
        "temperature":    0.0,    // Maximum determinism
        "repeat_last_n":  2,      // Tries to avoid repetitions
        "repeat_penalty": 1.8,    // Strong penalty for repetitions
        "top_k":          10,     // helps keep focus on most likely responses
        "top_p":          0.5,    // helps keep focus on most likely responses
    },
    Stream: &TRUE,  // Activates response streaming
}
```

Finally, the program sends the request and handles the response through streaming:

```go
err = client.Chat(ctx, req, func(resp api.ChatResponse) error {
    fmt.Print(resp.Message.Content)
    return nil
})
```

This part displays the response as it's generated by the LLM.

#### Parameters `top_k`, `top_p`, `repeat_last_n` and `repeat_penalty`

Before running the program, let's take a moment to discuss some of the request parameters:

- `top_k`: Reduces the probability of generating incoherent content. A higher value (e.g., 100) will give more diverse responses, while a lower value (e.g., 10) will be more conservative. (Default: 40)
- `top_p`: Works in conjunction with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)
- `repeat_last_n`: Determines how far back the model should look to avoid repetitions.
- `repeat_penalty`: Defines the intensity of repetition penalization. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more tolerant.

These parameters are essential for controlling the quality and coherence of responses generated by the model. But even though there are recommendations, it's important to adjust them according to your specific needs and especially according to the model (in my opinion).

### Let's ask that question!

So, let's run the program and see what "Baby Qwen" has to say about the biological compatibility of the human species in the Aethelgard universe.

```bash
go run main.go
```

And here's "Baby Qwen's" response:

```raw
The Human species is highly compatible with other species, including the Aethelgard. Humans are known for their intelligence, adaptability, and ability to communicate effectively with others. They are also known for their ability to adapt and evolve over time, making them a valuable and adaptable species.
```

Well well well, while the response seems correct, it's very generic, and ultimately I'm no more informed about the biological compatibility of humans in the Aethelgard universe.

**Warning, anthropomorphic remark ahead!**

In fact, I think "Baby Qwen" has trouble ingesting all the content and processing it correctly. So, it's not able to have a global vision of the Aethelgard universe and then ultimately keep only the relevant information to answer the question.

We'll therefore need to reduce the context provided to "Baby Qwen" with only information related to the question, so it can better process the information and respond more precisely to the question. 🤔

To put it another way: we need to extract from the document the information that talks about humans, and also those where we talk about both humans, other species, and biological compatibility. Put all this in a smaller and more precise context so that "Baby Qwen" can better focus.

This is where RAG (Retrieval Augmented Generation) comes into play. 🚀

## Let's talk about RAG

Doing RAG means searching for similarities in content based on the user's question.

The principle is as follows:

![rag](/imgs/rag.png)

RAG is there to overcome the limitations of LLMs.

For example, if you want to discuss your documentation via Ollama, the model will probably "confess" its ignorance on the subject.

This is where RAG (Retrieval Augmented Generation) comes in, once again, a solution to enrich the model's knowledge without having to retrain it.

The process takes place in several steps:

- Fragmentation: Your documents are cut into small coherent segments (called **chunks**).
- Vector Transformation: Each fragment (chunk) is converted into a "unique mathematical fingerprint" (a vector, also called **embeddings**) that captures the semantics of its content. **The calculation of embeddings is done by specialized LLMs capable of returning a set of vector coordinates from text**.
- Organization in Vector Database: These vectors are saved in a specialized database (we'll just work with in-memory data for now).

Then, when a user asks a question, the system will work as follows:

- The question is first transformed into a vector
- The system searches for the most relevant fragments (chunks) in its library. **In fact, since the fragments are linked to vectors, we'll be able to calculate the distances between vectors and keep only the closest ones**.
- These fragments are integrated into the context of the question
- The whole thing is transmitted to the language model which can then draw from this new knowledge to formulate a precise and relevant response (in theory).

### How do we fragment the document?

So, "chunking" is the process of breaking down a document into smaller coherent segments (chunks) to facilitate their processing by language models.

There are various cutting strategies, here are the main ones:

1. By fixed size
  - Cutting into segments of a fixed number of characters or tokens
  - Simple but risks cutting in the middle of sentences or ideas
2. By structure
  - Respecting natural divisions of the document (paragraphs, sections)
  - Better preserves context but can create chunks of uneven sizes
3. By meaning
  - Cutting while preserving coherent semantic units
  - More complex but gives better results for understanding

We should try to:

- Maintain an optimal chunk size (neither too big nor too small)
- **Ensure overlap between chunks to not lose context** (in the case of the 1st strategy)
- Preserve the logical structure of the document (rather strategies 2 and 3)
- Preserve important metadata (titles, references) to facilitate similarity search (we create sorts of links or bookmarks).

### How do we calculate embeddings?

We'll use Ollama and a specialized language model to calculate the embeddings of our chunks (We call these **embedding models**). Ollama provides an API for this, which allows transforming text into vectors. I'll let you read this blog post that explains it in detail: [Embedding models](https://ollama.com/blog/embedding-models).

For our first experiments, we'll use [`snowflake-arctic-embed:33m`](https://ollama.com/library/snowflake-arctic-embed:33m) as the embedding model. It's very small (67MB) but quite sufficient for our needs.

So don't forget to download it:

```bash
ollama pull snowflake-arctic-embed:33m
```

### How do we calculate distances between vectors?

To calculate distances between vectors, we'll do some coding and use the **"Cosine Similarity"** method.

In the context of embeddings (the vectors):

- **Vectors are close (or similar)** when the **cosine similarity** is **close to 1**.
- **Vectors are distant (or dissimilar)** when the **cosine similarity** is **close to 0** (or when the cosine distance is close to 1).

**Key Points**:

1. **Cosine Similarity** measures the angle between two vectors:
   - If the vectors are **aligned (pointing in the same direction)**, the similarity is `1`.
   - If the vectors are **perpendicular**, the similarity is `0`.
   - If the vectors are **opposite**, the similarity is `-1`.
2. **Cosine Distance** (it's the inverse of similarity):
   - It's simply `1 - cosine similarity`.
   - When vectors are very similar, the cosine distance will be close to `0`.
   - When vectors are very dissimilar, the cosine distance will be close to `1`.
3. **"Proximity" Threshold**:
   - There is no universal threshold, but typically:
     - **Cosine Similarity ≥ 0.8**: Vectors are considered close (very similar).
     - **Cosine Distance ≤ 0.2**: Vectors are considered close.

**So, for example**:

- If you calculate a cosine similarity of `0.95` between two embeddings, this means the vectors are very close and probably represent similar concepts.
- Conversely, a cosine **similarity** of `0.2` or a cosine **distance** of `0.8` indicates they are quite far apart.

> This is important. At first, I would reverse or mix up the concepts of similarity and distance.

**Practical Use**:
In the context of embeddings for tasks like retrieval-augmented generation (RAG):

- **Close vectors** often mean that the embeddings are semantically similar, meaning they probably represent related ideas or contexts.
- **Distance thresholds** are generally context-specific and may require experimentation to be finely tuned.

Yes, I know, that's a lot, but it's important for what follows. 🤓

Let's code now!

## Let's code the RAG

Once again, I'll start with the complete code and then explain the details. Today we'll use chunking strategy number 1, and we'll see in future blog posts other strategies to improve the relevance of responses.

```golang
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
```

### Explanations (grab yourself a coffee ☕️)

The program uses embeddings to find the most relevant parts of a document before answering a question.

Let's understand the overall structure of the program:

#### Important Data Structures:

I defined two key data structures to store embeddings and similarities:

```go
type VectorRecord struct {
    Prompt    string    `json:"prompt"`
    Embedding []float64 `json:"embedding"`
}

type Similarity struct {
    Prompt           string
    CosineSimilarity float64
}
```

These structures are essential for storing and comparing embeddings. VectorRecord associates a text (the chunk content) with its vector embedding, while Similarity stores a text and its calculated similarity with the question.

#### Utility Functions:

##### Chunking

The `ChunkText` function splits the text into fixed-size pieces with overlap:

```go
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
```

This function is important because it allows processing of long texts by breaking them into pieces that slightly overlap (overlap), which helps maintain context between chunks.

##### Embeddings

The `GetEmbeddingFromChunk` function converts text into a vector using Ollama's Go API (`EmbeddingRequest`):

```go
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
```

This function uses the **"snowflake-arctic-embed:33m"** model to transform text into a vector that captures its semantic meaning.

##### Similarities

The `CosineSimilarity` function calculates the similarity between two vectors:

```go
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
```

This function implements the cosine similarity measure, which provides a value between -1 and 1 indicating how similar two vectors are.

#### Now, the Main Program

First, the program loads and chunks the context:

```go
context, err := os.ReadFile("../content/chronicles-of-aethelgard.md")
chunks := ChunkText(string(context), 1024, 256)
```

The program reads the context file and splits it into pieces of 1024 characters with a 256-character overlap.

Next, we have the creation of embeddings:

```go
vectorStore := []VectorRecord{}
for idx, chunk := range chunks {
    embedding, _ := GetEmbeddingFromChunk(ctx, client, chunk)
    record := VectorRecord{
        Prompt:    chunk,
        Embedding: embedding,
    }
    vectorStore = append(vectorStore, record)
}
```

Each piece of text is converted into an embedding and stored in memory in an array or slice of `VectorRecord`.

We can now launch the similarity search:

```go
embeddingFromQuestion, _ := GetEmbeddingFromChunk(ctx, client, question)

// Search similarities between the question and the vectors of the store
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
```

The program calculates the similarity between the question and each piece of text, then selects the 5 most relevant pieces.

```go
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
}
```

**Now we can move on to generating the response**:

The program then uses these 5 most relevant pieces to build a new, more focused context, which is provided to the language model along with the question:

```go
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
```

And then we can launch our request as before.

Let's go!

### Let's run the program

```bash
go run main.go
```

The program will cut the document into about a hundred chunks, then calculate the embeddings for each chunk. Then, it will calculate the similarity between the question and each chunk. It will then select the 5 chunks most similar to the question, concatenate them to form a new more targeted context, and finally ask "Baby Qwen" the question.

In my trials, the 5 most similar chunks had a cosine similarity of `0.80` to `0.84`. That's a good score, but it can be adjusted. You need to experiment to find the right similarity threshold and also try with other LLMs that calculate embeddings.

And here's "Baby Qwen's" response:

```raw
The Human species is characterized by its unique genetic flexibility, which allows it to produce viable offspring with various humanoid species. This adaptability has profound implications

- Half-Evans: The most common mixed heritage, often serving as bridges between human and elven communities.
- Half-orcs: They emerge from either peaceful unions or historical conflicts, facing varying degrees of acceptance.
- Rarer combinations (human-dwarf, human-halfling) occur but typically require magical intervention.

This biological adaptability influences human society, leading to complex inheritance laws and social structures accommodating mixed-heritage individuals.
```

This is much better! The response is more precise and detailed, and it addresses specific aspects of humans' biological compatibility with other species in the Aethelgard universe.

Now, we could see if taking more results, like the top 10, makes the response even more precise. That's up to you to try!

Today we've covered the basics to start experimenting with RAG. In future blog posts, we'll explore other chunking strategies and other language models, as well as some concepts about vector databases for storing embeddings.

See you soon for new adventures! 🚀

Of course, you'll find the complete code on [ollama-tlms-golang/02-rag-first-contact](https://github.com/ollama-tlms-golang/02-rag-first-contact)