package main

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

// Estructura del estado del job
type JobState struct {
	Status        string    `json:"status"`                  // queued, processing, completed, failed
	Transcription string    `json:"transcription,omitempty"` // puede incluir letras yorùbá
	Translation   string    `json:"translation,omitempty"`
	Error         string    `json:"error,omitempty"`
	Timestamp     time.Time `json:"timestamp"`
}

// Entrada del cliente
type RequestBody struct {
	URL       string `json:"url"`
	Language  string `json:"language"`
	Translate bool   `json:"translate"`
}

// Petición al microservicio Python
type PythonRequest struct {
	URL       string `json:"url"`
	Language  string `json:"language"`
	Translate bool   `json:"translate"`
}

var jobStore = make(map[string]*JobState)
var mu sync.RWMutex

func main() {
	router := gin.Default()

	// ✅ Listar todos los jobs
	router.GET("/jobs", func(c *gin.Context) {
		mu.RLock()
		defer mu.RUnlock()

		response := make(map[string]*JobState)
		for id, job := range jobStore {
			response[id] = job
		}
		c.Header("Content-Type", "application/json; charset=utf-8")
		c.JSON(http.StatusOK, response)
	})

	// ✅ Crear un nuevo job asincrónico
	router.POST("/process", func(c *gin.Context) {
		var input RequestBody
		if err := c.ShouldBindJSON(&input); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		jobID := uuid.NewString()
		mu.Lock()
		jobStore[jobID] = &JobState{
			Status:    "queued",
			Timestamp: time.Now(),
		}
		mu.Unlock()

		go processJob(jobID, input)

		c.Header("Content-Type", "application/json; charset=utf-8")
		c.JSON(http.StatusAccepted, gin.H{
			"job_id": jobID,
			"status": "queued",
		})
	})

	// ✅ Obtener resultado de un job por ID
	router.GET("/result/:job_id", func(c *gin.Context) {
		jobID := c.Param("job_id")

		mu.RLock()
		job, exists := jobStore[jobID]
		mu.RUnlock()

		if !exists {
			c.JSON(http.StatusNotFound, gin.H{"error": "job not found"})
			return
		}

		c.Header("Content-Type", "application/json; charset=utf-8")
		c.JSON(http.StatusOK, job)
	})

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	log.Printf("🚀 API corriendo en http://localhost:%s", port)
	router.Run(":" + port)
}

// Ejecuta el trabajo en background
func processJob(jobID string, input RequestBody) {
	mu.Lock()
	jobStore[jobID].Status = "processing"
	mu.Unlock()

	payload := PythonRequest{
		URL:       input.URL,
		Language:  input.Language,
		Translate: input.Translate,
	}
	jsonData, _ := json.Marshal(payload)

	resp, err := http.Post("http://whisper_service:8000/transcribe", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		mu.Lock()
		jobStore[jobID].Status = "failed"
		jobStore[jobID].Error = err.Error()
		mu.Unlock()
		return
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	var result map[string]string
	json.Unmarshal(body, &result)

	mu.Lock()
	defer mu.Unlock()

	if resp.StatusCode != http.StatusOK {
		jobStore[jobID].Status = "failed"
		jobStore[jobID].Error = string(body)
		return
	}

	jobStore[jobID].Status = "completed"
	jobStore[jobID].Transcription = result["transcription"]
	jobStore[jobID].Translation = result["translation"]
}
