// SPDX-License-Identifier: AGPL-3.0-or-later

package vectorstore

import (
	"testing"

	"github.com/asciimoo/hister/server/testutil"
)

func newTestSQLiteVectorStore(t *testing.T) *sqliteVectorStore {
	t.Helper()

	cfg := testutil.Config(t)
	cfg.Server.Database = "hister.sqlite3"
	cfg.SemanticSearch.Dimensions = 2

	store, err := newSQLite(cfg)
	if err != nil {
		t.Fatalf("newSQLite returned an error: %v", err)
	}
	if err := store.Init(); err != nil {
		_ = store.Close()
		t.Fatalf("Init returned an error: %v", err)
	}
	live := store.(*sqliteVectorStore)
	t.Cleanup(func() {
		if err := live.Close(); err != nil {
			t.Errorf("close SQLite vector store: %v", err)
		}
	})
	return live
}

func requireSQLiteResultDoc(t *testing.T, store *sqliteVectorStore, query []float32, want string) {
	t.Helper()

	results, err := store.Search(query, 10, 0.99, 0)
	if err != nil {
		t.Fatalf("Search returned an error: %v", err)
	}
	if len(results) != 1 || results[0].DocID != want {
		t.Fatalf("Search returned %#v, want only document %q", results, want)
	}
}

func requireSQLiteNoResults(t *testing.T, store *sqliteVectorStore, query []float32) {
	t.Helper()

	results, err := store.Search(query, 10, 0.99, 0)
	if err != nil {
		t.Fatalf("Search returned an error: %v", err)
	}
	if len(results) != 0 {
		t.Fatalf("Search returned %#v, want no results", results)
	}
}

func TestSQLiteReindexRollbackPreservesLiveVectors(t *testing.T) {
	live := newTestSQLiteVectorStore(t)
	if err := live.PutChunks("old", 0, []Chunk{{Index: 0, Text: "old", Embedding: []float32{1, 0}}}); err != nil {
		t.Fatalf("seed live vector: %v", err)
	}

	staged, err := live.BeginReindex()
	if err != nil {
		t.Fatalf("BeginReindex returned an error: %v", err)
	}
	if err := staged.PutChunks("new", 0, []Chunk{{Index: 0, Text: "new", Embedding: []float32{0, 1}}}); err != nil {
		t.Fatalf("seed staged vector: %v", err)
	}
	if err := staged.Rollback(); err != nil {
		t.Fatalf("Rollback returned an error: %v", err)
	}

	requireSQLiteResultDoc(t, live, []float32{1, 0}, "old")
	requireSQLiteNoResults(t, live, []float32{0, 1})
}

func TestSQLiteReindexCommitReplacesLiveVectors(t *testing.T) {
	live := newTestSQLiteVectorStore(t)
	if err := live.PutChunks("old", 0, []Chunk{{Index: 0, Text: "old", Embedding: []float32{1, 0}}}); err != nil {
		t.Fatalf("seed live vector: %v", err)
	}

	staged, err := live.BeginReindex()
	if err != nil {
		t.Fatalf("BeginReindex returned an error: %v", err)
	}
	if err := staged.PutChunks("new", 0, []Chunk{{Index: 0, Text: "new", Embedding: []float32{0, 1}}}); err != nil {
		t.Fatalf("seed staged vector: %v", err)
	}
	if err := staged.Commit(); err != nil {
		t.Fatalf("Commit returned an error: %v", err)
	}

	requireSQLiteResultDoc(t, live, []float32{0, 1}, "new")
	requireSQLiteNoResults(t, live, []float32{1, 0})
}
