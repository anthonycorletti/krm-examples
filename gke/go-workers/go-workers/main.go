package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"sync"
	"time"
)

type Repository struct {
	ID          int    `json:"id"`
	Name        string `json:"name"`
	FullName    string `json:"full_name"`
	HTMLURL     string `json:"html_url"`
	Description string `json:"description"`
	Language    string `json:"language"`
	Stars       int    `json:"stargazers_count"`
	Forks       int    `json:"forks_count"`
	CreatedAt   string `json:"created_at"`
	UpdatedAt   string `json:"updated_at"`
}

type GitHubSearchResponse struct {
	Items []Repository `json:"items"`
}

type RepoContent struct {
	Name        string `json:"name"`
	Path        string `json:"path"`
	SHA         string `json:"sha"`
	Size        int    `json:"size"`
	URL         string `json:"url"`
	HTMLURL     string `json:"html_url"`
	GitURL      string `json:"git_url"`
	DownloadURL string `json:"download_url"`
	Type        string `json:"type"`
	Content     string `json:"content,omitempty"`
	Encoding    string `json:"encoding,omitempty"`
}

type ScrapingResult struct {
	Repository Repository    `json:"repository"`
	Contents   []RepoContent `json:"contents,omitempty"`
	Success    bool          `json:"success"`
	Error      string        `json:"error,omitempty"`
	ProcessID  int           `json:"process_id"`
}

func main() {
	http.HandleFunc("/scrape-trending", scrapeTrendingHandler)
	http.HandleFunc("/health", healthHandler)

	log.Println("Starting GitHub API scraper server on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	fmt.Fprint(w, "OK")
}

func scrapeTrendingHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	log.Println("Starting to fetch GitHub trending repositories via API...")

	// Fetch trending repositories using GitHub Search API
	repos, err := fetchTrendingRepos()
	if err != nil {
		log.Printf("Error fetching trending repos: %v", err)
		http.Error(w, "Failed to fetch trending repositories", http.StatusInternalServerError)
		return
	}

	log.Printf("Found %d trending repositories, starting scraping processes...", len(repos))

	// Channel to collect results
	results := make(chan ScrapingResult, len(repos))
	var wg sync.WaitGroup

	// Launch a goroutine for each repository
	for i, repo := range repos {
		wg.Add(1)
		go scrapeRepository(repo, i+1, results, &wg)
	}

	// Wait for all goroutines to complete
	go func() {
		wg.Wait()
		close(results)
	}()

	// Collect and log results
	var allResults []ScrapingResult
	successCount := 0

	for result := range results {
		allResults = append(allResults, result)

		if result.Success {
			successCount++
			log.Printf("SUCCESS [Process %d]: Scraped %s (%s) - %d stars, %d files found",
				result.ProcessID,
				result.Repository.FullName,
				result.Repository.Language,
				result.Repository.Stars,
				len(result.Contents))
		} else {
			log.Printf("ERROR [Process %d]: Failed to scrape %s - %s",
				result.ProcessID,
				result.Repository.FullName,
				result.Error)
		}
	}

	log.Printf("Completed scraping %d repositories (%d successful, %d failed)",
		len(allResults), successCount, len(allResults)-successCount)

	// Return results as JSON
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"message":     "Scraping completed",
		"total_repos": len(allResults),
		"successful":  successCount,
		"failed":      len(allResults) - successCount,
		"results":     allResults,
	})
}

func fetchTrendingRepos() ([]Repository, error) {
	// Use GitHub Search API to find trending repositories
	// Search for repositories created in the last week, sorted by stars
	query := url.QueryEscape("created:>" + time.Now().AddDate(0, 0, -7).Format("2006-01-02"))
	apiURL := fmt.Sprintf("https://api.github.com/search/repositories?q=%s&sort=stars&order=desc&per_page=15", query)

	client := &http.Client{
		Timeout: 30 * time.Second,
	}

	req, err := http.NewRequest("GET", apiURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set User-Agent header (required by GitHub API)
	req.Header.Set("User-Agent", "GitHub-Trending-Scraper/1.0")
	req.Header.Set("Accept", "application/vnd.github.v3+json")

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch trending repositories: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("GitHub API returned status %d: %s", resp.StatusCode, string(body))
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	var searchResponse GitHubSearchResponse
	if err := json.Unmarshal(body, &searchResponse); err != nil {
		return nil, fmt.Errorf("failed to parse JSON response: %w", err)
	}

	return searchResponse.Items, nil
}

func scrapeRepository(repo Repository, processID int, results chan<- ScrapingResult, wg *sync.WaitGroup) {
	defer wg.Done()

	log.Printf("[Process %d] Started scraping repository: %s", processID, repo.FullName)

	client := &http.Client{
		Timeout: 20 * time.Second,
	}

	// Fetch repository contents (root directory)
	contentsURL := fmt.Sprintf("https://api.github.com/repos/%s/contents", repo.FullName)

	req, err := http.NewRequest("GET", contentsURL, nil)
	if err != nil {
		results <- ScrapingResult{
			Repository: repo,
			Success:    false,
			Error:      fmt.Sprintf("Failed to create contents request: %v", err),
			ProcessID:  processID,
		}
		return
	}

	// Set headers for GitHub API
	req.Header.Set("User-Agent", "GitHub-Trending-Scraper/1.0")
	req.Header.Set("Accept", "application/vnd.github.v3+json")

	resp, err := client.Do(req)
	if err != nil {
		results <- ScrapingResult{
			Repository: repo,
			Success:    false,
			Error:      fmt.Sprintf("Failed to fetch repository contents: %v", err),
			ProcessID:  processID,
		}
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		results <- ScrapingResult{
			Repository: repo,
			Success:    false,
			Error:      fmt.Sprintf("GitHub API returned status %d: %s", resp.StatusCode, string(body)),
			ProcessID:  processID,
		}
		return
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		results <- ScrapingResult{
			Repository: repo,
			Success:    false,
			Error:      fmt.Sprintf("Failed to read contents response: %v", err),
			ProcessID:  processID,
		}
		return
	}

	var contents []RepoContent
	if err := json.Unmarshal(body, &contents); err != nil {
		results <- ScrapingResult{
			Repository: repo,
			Success:    false,
			Error:      fmt.Sprintf("Failed to parse contents JSON: %v", err),
			ProcessID:  processID,
		}
		return
	}

	log.Printf("[Process %d] Successfully scraped %s - found %d items in root directory",
		processID, repo.FullName, len(contents))

	results <- ScrapingResult{
		Repository: repo,
		Contents:   contents,
		Success:    true,
		ProcessID:  processID,
	}
}
