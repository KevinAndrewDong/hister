// SPDX-License-Identifier: AGPL-3.0-or-later

package indexer

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/asciimoo/hister/config"
	"github.com/asciimoo/hister/server/document"
	"github.com/asciimoo/hister/server/testutil"
	"github.com/asciimoo/hister/server/vectorstore"
)

type commitFailingVectorStore struct {
	vectorstore.VectorStore
	err error
}

func (s *commitFailingVectorStore) BeginReindex() (vectorstore.ReindexStore, error) {
	staged, err := s.VectorStore.BeginReindex()
	if err != nil {
		return nil, err
	}
	return &commitFailingReindexStore{ReindexStore: staged, err: s.err}, nil
}

type commitFailingReindexStore struct {
	vectorstore.ReindexStore
	err error
}

func (s *commitFailingReindexStore) Commit() error {
	return s.err
}

func TestConcurrentReindexReturnsAlreadyRunning(t *testing.T) {
	cfg := testutil.Config(t)
	idx, err := initializeIndexer(cfg.FullPath(""), false, false, "")
	if err != nil {
		t.Fatalf("initializeIndexer returned an error: %v", err)
	}

	idx.lifecycleMu.Lock()
	lifecycleLocked := true
	defer func() {
		if lifecycleLocked {
			idx.lifecycleMu.Unlock()
		}
		idx.Close()
	}()

	firstDone := make(chan error, 1)
	go func() {
		firstDone <- idx.ReindexContext(context.Background(), &config.Rules{}, false, false, false, nil)
	}()
	deadline := time.Now().Add(time.Second)
	for !idx.reindexInProgress.Load() && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if !idx.reindexInProgress.Load() {
		t.Fatal("first reindex did not claim the in-progress flag")
	}

	if err := idx.ReindexContext(context.Background(), &config.Rules{}, false, false, false, nil); err == nil || err.Error() != "Reindex is already running" {
		t.Fatalf("second ReindexContext error = %v, want already running", err)
	}

	idx.lifecycleMu.Unlock()
	lifecycleLocked = false
	if err := <-firstDone; err != nil {
		t.Fatalf("first ReindexContext returned an error: %v", err)
	}
}

func TestReindexAlreadyRunningDoesNotReadMetadata(t *testing.T) {
	cfg := testutil.Config(t)
	idx, err := initializeIndexer(cfg.FullPath(""), false, false, "")
	if err != nil {
		t.Fatalf("initializeIndexer returned an error: %v", err)
	}
	idx.Close()
	idx.reindexInProgress.Store(true)

	err = idx.ReindexContext(context.Background(), &config.Rules{}, false, false, false, nil)
	if err == nil || err.Error() != "Reindex is already running" {
		t.Fatalf("ReindexContext error = %v, want already running without reading closed metadata", err)
	}
}

func TestStopEmbeddingQueueWaitsForMutationLock(t *testing.T) {
	_, cancel := context.WithCancel(context.Background())
	idx := &Indexer{embeddingQueue: &embeddingQueue{cancel: cancel}}

	idx.reindexMu.RLock()
	mutationLocked := true
	defer func() {
		if mutationLocked {
			idx.reindexMu.RUnlock()
		}
	}()
	started := make(chan struct{})
	done := make(chan struct{})
	go func() {
		close(started)
		idx.stopEmbeddingQueue()
		close(done)
	}()
	<-started

	select {
	case <-done:
		t.Fatal("embedding queue stopped while a mutation held reindexMu")
	case <-time.After(50 * time.Millisecond):
	}
	if idx.embeddingQueue == nil {
		t.Fatal("embedding queue was detached before the mutation completed")
	}

	idx.reindexMu.RUnlock()
	mutationLocked = false
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("embedding queue did not stop after the mutation completed")
	}
}

func TestStopEmbeddingQueueStaysAttachedUntilWorkersStop(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	workerCanceled := make(chan struct{})
	releaseWorker := make(chan struct{})
	queue := &embeddingQueue{ctx: ctx, cancel: cancel}
	queue.wg.Go(func() {
		<-ctx.Done()
		close(workerCanceled)
		<-releaseWorker
	})
	idx := &Indexer{embeddingQueue: queue}
	done := make(chan struct{})
	go func() {
		idx.stopEmbeddingQueue()
		close(done)
	}()

	select {
	case <-workerCanceled:
	case <-time.After(time.Second):
		t.Fatal("embedding queue workers were not canceled")
	}
	if idx.embeddingQueue != queue {
		t.Fatal("embedding queue was detached before its workers stopped")
	}

	close(releaseWorker)
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("embedding queue did not detach after its workers stopped")
	}
	if idx.embeddingQueue != nil {
		t.Fatal("embedding queue remained attached after its workers stopped")
	}
}

func TestReindexVectorCommitFailureRestoresKeywordAndVectorIndexes(t *testing.T) {
	cfg := testutil.Config(t)
	cfg.SemanticSearch.EmbeddingEndpoint = "http://127.0.0.1:0"
	cfg.SemanticSearch.EmbeddingModel = "test-model"
	cfg.SemanticSearch.Dimensions = 2

	idx, err := initializeIndexer(cfg.FullPath(""), false, false, "")
	if err != nil {
		t.Fatalf("initializeIndexer returned an error: %v", err)
	}
	defer idx.Close()

	store, err := vectorstore.New(cfg)
	if err != nil {
		t.Fatalf("vectorstore.New returned an error: %v", err)
	}
	if err := store.Init(); err != nil {
		_ = store.Close()
		t.Fatalf("vector store Init returned an error: %v", err)
	}
	commitErr := errors.New("forced vector commit failure")
	idx.vectorStore = &commitFailingVectorStore{VectorStore: store, err: commitErr}
	idx.embedder = vectorstore.NewEmbedder(&cfg.SemanticSearch)

	doc := &document.Document{
		URL:       "https://example.com/reindex-commit-failure",
		Title:     "Reindex commit failure",
		Text:      "Both live indexes must survive a failed vector commit.",
		Language:  document.UnknownLanguage,
		Processed: true,
		AddCount:  1,
	}
	if err := idx.save(doc); err != nil {
		t.Fatalf("seed indexed document: %v", err)
	}
	if err := store.PutChunks(doc.ID(), 0, []vectorstore.Chunk{
		{Index: 0, Text: "old", Embedding: []float32{1, 0}},
	}); err != nil {
		t.Fatalf("seed live vector: %v", err)
	}

	rules := &config.Rules{Skip: &config.Rule{ReStrs: []string{doc.URL}}}
	reindexErr := idx.ReindexContext(context.Background(), rules, false, false, false, nil)
	if !errors.Is(reindexErr, commitErr) {
		t.Fatalf("ReindexContext error = %v, want %v", reindexErr, commitErr)
	}
	if idx.GetByURLAndUser(doc.URL, 0) == nil {
		t.Fatal("keyword index published the replacement after vector commit failure")
	}
	results, err := idx.vectorStore.Search([]float32{1, 0}, 10, 0.99, 0)
	if err != nil {
		t.Fatalf("search live vectors: %v", err)
	}
	if len(results) != 1 || results[0].DocID != doc.ID() {
		t.Fatalf("live vector search returned %#v, want document %q", results, doc.ID())
	}
}

func TestReindexEmbeddingFailurePreservesLiveVectors(t *testing.T) {
	var requests atomic.Int32
	embeddingServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		http.Error(w, "embedding service unavailable", http.StatusServiceUnavailable)
	}))
	defer embeddingServer.Close()

	cfg := testutil.Config(t)
	cfg.SemanticSearch.EmbeddingEndpoint = embeddingServer.URL
	cfg.SemanticSearch.EmbeddingModel = "test-model"
	cfg.SemanticSearch.Dimensions = 2
	cfg.SemanticSearch.MaxContextLength = 32
	cfg.SemanticSearch.MaxEmbeddingBatchSize = 1

	idx, err := initializeIndexer(cfg.FullPath(""), false, false, "")
	if err != nil {
		t.Fatalf("initializeIndexer returned an error: %v", err)
	}
	defer idx.Close()

	store, err := vectorstore.New(cfg)
	if err != nil {
		t.Fatalf("vectorstore.New returned an error: %v", err)
	}
	if err := store.Init(); err != nil {
		_ = store.Close()
		t.Fatalf("vector store Init returned an error: %v", err)
	}
	idx.vectorStore = store
	idx.embedder = vectorstore.NewEmbedder(&cfg.SemanticSearch)

	doc := &document.Document{
		URL:       "https://example.com/reindex-embedding-failure",
		Title:     "Reindex failure",
		Text:      "The live vector must survive a failed reindex.",
		Language:  document.UnknownLanguage,
		Processed: true,
		AddCount:  1,
	}
	if err := idx.save(doc); err != nil {
		t.Fatalf("seed indexed document: %v", err)
	}
	if err := store.PutChunks(doc.ID(), 0, []vectorstore.Chunk{
		{Index: 0, Text: "old", Embedding: []float32{1, 0}},
	}); err != nil {
		t.Fatalf("seed live vector: %v", err)
	}

	reindexErr := idx.ReindexContext(context.Background(), &config.Rules{}, false, false, false, nil)
	if reindexErr == nil {
		t.Error("ReindexContext succeeded despite embedding failure")
	}
	if requests.Load() == 0 {
		t.Fatal("reindex did not call the embedding endpoint")
	}

	results, err := idx.vectorStore.Search([]float32{1, 0}, 10, 0.99, 0)
	if err != nil {
		t.Fatalf("search live vectors: %v", err)
	}
	if len(results) != 1 || results[0].DocID != doc.ID() {
		t.Fatalf("live vector search returned %#v, want document %q", results, doc.ID())
	}
}

func TestReindexCancellationPreservesLiveVectors(t *testing.T) {
	requestStarted := make(chan struct{})
	releaseEmbedding := make(chan struct{})
	var startOnce sync.Once
	var releaseOnce sync.Once
	embeddingServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		startOnce.Do(func() { close(requestStarted) })
		<-releaseEmbedding

		var request struct {
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		data := make([]map[string]any, len(request.Input))
		for i := range data {
			data[i] = map[string]any{"embedding": []float64{1, 0}}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"data": data})
	}))
	release := func() { releaseOnce.Do(func() { close(releaseEmbedding) }) }
	defer func() {
		release()
		embeddingServer.Close()
	}()

	cfg := testutil.Config(t)
	cfg.SemanticSearch.EmbeddingEndpoint = embeddingServer.URL
	cfg.SemanticSearch.EmbeddingModel = "test-model"
	cfg.SemanticSearch.Dimensions = 2
	cfg.SemanticSearch.MaxContextLength = 32

	idx, err := initializeIndexer(cfg.FullPath(""), false, false, "")
	if err != nil {
		t.Fatalf("initializeIndexer returned an error: %v", err)
	}
	defer idx.Close()

	store, err := vectorstore.New(cfg)
	if err != nil {
		t.Fatalf("vectorstore.New returned an error: %v", err)
	}
	if err := store.Init(); err != nil {
		_ = store.Close()
		t.Fatalf("vector store Init returned an error: %v", err)
	}
	idx.vectorStore = store
	idx.embedder = vectorstore.NewEmbedder(&cfg.SemanticSearch)

	doc := &document.Document{
		URL:       "https://example.com/reindex-cancellation",
		Title:     "Reindex cancellation",
		Text:      "The live vector must survive cancellation during reindex.",
		Language:  document.UnknownLanguage,
		Processed: true,
		AddCount:  1,
	}
	if err := idx.save(doc); err != nil {
		t.Fatalf("seed indexed document: %v", err)
	}
	if err := store.PutChunks(doc.ID(), 0, []vectorstore.Chunk{
		{Index: 0, Text: "old", Embedding: []float32{1, 0}},
	}); err != nil {
		t.Fatalf("seed live vector: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	reindexDone := make(chan error, 1)
	go func() {
		reindexDone <- idx.ReindexContext(ctx, &config.Rules{}, false, false, false, nil)
	}()
	select {
	case <-requestStarted:
	case <-time.After(5 * time.Second):
		t.Fatal("reindex did not call the embedding endpoint")
	}
	cancel()
	release()

	select {
	case err := <-reindexDone:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("ReindexContext error = %v, want context.Canceled", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("ReindexContext did not return after cancellation")
	}

	results, err := idx.vectorStore.Search([]float32{1, 0}, 10, 0.99, 0)
	if err != nil {
		t.Fatalf("search live vectors: %v", err)
	}
	if len(results) != 1 || results[0].DocID != doc.ID() {
		t.Fatalf("live vector search returned %#v, want document %q", results, doc.ID())
	}
}
