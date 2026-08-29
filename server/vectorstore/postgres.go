// SPDX-License-Identifier: AGPL-3.0-or-later

package vectorstore

import (
	"database/sql"
	"errors"
	"fmt"
	"strings"

	"github.com/asciimoo/hister/config"

	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/rs/zerolog/log"
)

type pgVectorStore struct {
	db          *sql.DB
	dimensions  int
	tableName   string
	indexPrefix string
	staging     bool
	ownsDB      bool
}

const (
	pgEmbeddingsTable = "embeddings"
)

func newPostgres(cfg *config.Config) (VectorStore, error) {
	_, dsn := cfg.DatabaseConnection()
	db, err := sql.Open("pgx", dsn)
	if err != nil {
		return nil, fmt.Errorf("open postgres for vectors: %w", err)
	}
	return &pgVectorStore{
		db:          db,
		dimensions:  cfg.SemanticSearch.Dimensions,
		tableName:   pgEmbeddingsTable,
		indexPrefix: pgEmbeddingsTable,
		ownsDB:      true,
	}, nil
}

func pgIdentifier(name string) string {
	return `"` + strings.ReplaceAll(name, `"`, `""`) + `"`
}

func (p *pgVectorStore) Init() error {
	if _, err := p.db.Exec(`CREATE EXTENSION IF NOT EXISTS vector`); err != nil {
		return fmt.Errorf("create pgvector extension: %w", err)
	}
	log.Info().Msg("pgvector extension enabled")

	if err := p.initSchema(); err != nil {
		return err
	}
	return nil
}

func (p *pgVectorStore) initSchema() error {
	stmt := fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (
		chunk_key TEXT PRIMARY KEY,
		doc_id TEXT NOT NULL,
		chunk_idx INTEGER NOT NULL DEFAULT 0,
		user_id INTEGER NOT NULL DEFAULT 0,
		chunk_text TEXT NOT NULL DEFAULT '',
		embedding vector(%d)
	)`, pgIdentifier(p.tableName), p.dimensions)
	if _, err := p.db.Exec(stmt); err != nil {
		return fmt.Errorf("create embeddings table: %w", err)
	}

	// HNSW index for cosine distance.
	_, err := p.db.Exec(fmt.Sprintf(`CREATE INDEX IF NOT EXISTS %s
		ON %s USING hnsw (embedding vector_cosine_ops)`,
		pgIdentifier(p.indexPrefix+"_hnsw_idx"), pgIdentifier(p.tableName)))
	if err != nil {
		return fmt.Errorf("create HNSW index: %w", err)
	}
	if _, err := p.db.Exec(fmt.Sprintf(`CREATE INDEX IF NOT EXISTS %s ON %s (user_id)`,
		pgIdentifier(p.indexPrefix+"_user_idx"), pgIdentifier(p.tableName))); err != nil {
		return fmt.Errorf("create user_id index: %w", err)
	}
	if _, err := p.db.Exec(fmt.Sprintf(`CREATE INDEX IF NOT EXISTS %s ON %s (doc_id)`,
		pgIdentifier(p.indexPrefix+"_doc_idx"), pgIdentifier(p.tableName))); err != nil {
		return fmt.Errorf("create doc_id index: %w", err)
	}
	return nil
}

func (p *pgVectorStore) BeginReindex() (ReindexStore, error) {
	if p.staging {
		return nil, errors.New("cannot start a reindex from a staging vector store")
	}
	suffix := nextReindexSuffix()
	stage := &pgVectorStore{
		db:          p.db,
		dimensions:  p.dimensions,
		tableName:   p.tableName + suffix,
		indexPrefix: p.indexPrefix + suffix,
		staging:     true,
	}
	if _, err := p.db.Exec(fmt.Sprintf(`DROP TABLE IF EXISTS %s`, pgIdentifier(stage.tableName))); err != nil {
		return nil, fmt.Errorf("remove previous reindex table: %w", err)
	}
	if err := stage.initSchema(); err != nil {
		_ = stage.Rollback()
		return nil, fmt.Errorf("create reindex vector store: %w", err)
	}
	return stage, nil
}

func (p *pgVectorStore) PutChunks(docID string, userID uint, chunks []Chunk) error {
	if len(chunks) == 0 {
		return nil
	}
	tx, err := p.db.Begin()
	if err != nil {
		return fmt.Errorf("begin transaction: %w", err)
	}
	defer func() {
		if err != nil {
			_ = tx.Rollback()
		}
	}()

	// Delete all existing chunks for this document.
	if _, err = tx.Exec(fmt.Sprintf(`DELETE FROM %s WHERE doc_id = $1`, pgIdentifier(p.tableName)), docID); err != nil {
		return fmt.Errorf("delete old embeddings: %w", err)
	}

	// Build a single multi-row INSERT for all chunks.
	const cols = 6 // chunk_key, doc_id, chunk_idx, user_id, chunk_text, embedding
	var sb strings.Builder
	fmt.Fprintf(&sb, `INSERT INTO %s(chunk_key, doc_id, chunk_idx, user_id, chunk_text, embedding) VALUES `, pgIdentifier(p.tableName))
	args := make([]any, 0, len(chunks)*cols)
	for i, c := range chunks {
		if i > 0 {
			sb.WriteString(", ")
		}
		base := i*cols + 1
		fmt.Fprintf(&sb, "($%d, $%d, $%d, $%d, $%d, $%d)", base, base+1, base+2, base+3, base+4, base+5)
		args = append(args, chunkKey(docID, c.Index), docID, c.Index, userID, c.Text, pgVectorLiteral(c.Embedding))
	}
	if _, err = tx.Exec(sb.String(), args...); err != nil {
		return fmt.Errorf("insert embedding chunks: %w", err)
	}

	if err = tx.Commit(); err != nil {
		return fmt.Errorf("commit chunks: %w", err)
	}
	return nil
}

func (p *pgVectorStore) Delete(docID string) error {
	_, err := p.db.Exec(fmt.Sprintf(`DELETE FROM %s WHERE doc_id = $1`, pgIdentifier(p.tableName)), docID)
	if err != nil {
		return fmt.Errorf("delete embeddings: %w", err)
	}
	return nil
}

func (p *pgVectorStore) Search(vector []float32, topK int, threshold float64, userID uint) (_ []Result, err error) {
	candidateLimit := searchCandidateLimit(topK)
	userResults, err := p.searchUser(vector, candidateLimit, threshold, userID)
	if err != nil || userID == 0 {
		return diversifySearchResults(userResults, topK, maxChunksPerDocument), err
	}
	globalResults, err := p.searchUser(vector, candidateLimit, threshold, 0)
	if err != nil {
		return nil, err
	}
	merged := mergeSearchResults(0, userResults, globalResults)
	return diversifySearchResults(merged, topK, maxChunksPerDocument), nil
}

func (p *pgVectorStore) searchUser(vector []float32, topK int, threshold float64, userID uint) (_ []Result, err error) {
	vecStr := pgVectorLiteral(vector)
	rows, err := p.db.Query(fmt.Sprintf(
		`SELECT doc_id, chunk_idx, chunk_text, 1 - (embedding <=> $1::vector) AS similarity
			 FROM %s
			 WHERE 1 - (embedding <=> $1::vector) >= $2
			   AND user_id = $4
			 ORDER BY embedding <=> $1::vector
			 LIMIT $3`, pgIdentifier(p.tableName)),
		vecStr, threshold, topK, userID,
	)
	if err != nil {
		return nil, fmt.Errorf("vector search for user %d: %w", userID, err)
	}
	defer func() {
		if cerr := rows.Close(); err == nil {
			err = cerr
		}
	}()

	var results []Result
	for rows.Next() {
		var r Result
		if err := rows.Scan(&r.DocID, &r.ChunkIdx, &r.ChunkText, &r.Similarity); err != nil {
			return nil, fmt.Errorf("scan vector result: %w", err)
		}
		results = append(results, r)
	}
	return results, rows.Err()
}

func (p *pgVectorStore) Clear() error {
	if _, err := p.db.Exec(fmt.Sprintf(`DELETE FROM %s`, pgIdentifier(p.tableName))); err != nil {
		return fmt.Errorf("clear embeddings: %w", err)
	}
	return nil
}

func (p *pgVectorStore) Commit() error {
	if !p.staging {
		return errors.New("cannot commit a live vector store")
	}
	tx, err := p.db.Begin()
	if err != nil {
		return fmt.Errorf("begin vector store reindex commit: %w", err)
	}
	defer func() {
		_ = tx.Rollback()
	}()
	if _, err := tx.Exec(fmt.Sprintf(`DROP TABLE %s`, pgIdentifier(pgEmbeddingsTable))); err != nil {
		return fmt.Errorf("drop old embeddings table: %w", err)
	}
	if _, err := tx.Exec(fmt.Sprintf(`ALTER TABLE %s RENAME TO %s`,
		pgIdentifier(p.tableName), pgIdentifier(pgEmbeddingsTable))); err != nil {
		return fmt.Errorf("promote embeddings table: %w", err)
	}
	for _, suffix := range []string{"hnsw_idx", "user_idx", "doc_idx"} {
		if _, err := tx.Exec(fmt.Sprintf(`ALTER INDEX %s RENAME TO %s`,
			pgIdentifier(p.indexPrefix+"_"+suffix), pgIdentifier(pgEmbeddingsTable+"_"+suffix))); err != nil {
			return fmt.Errorf("promote embeddings index: %w", err)
		}
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit vector store reindex: %w", err)
	}
	p.tableName = pgEmbeddingsTable
	p.indexPrefix = pgEmbeddingsTable
	p.staging = false
	p.ownsDB = true
	return nil
}

func (p *pgVectorStore) Rollback() error {
	if !p.staging {
		return nil
	}
	if _, err := p.db.Exec(fmt.Sprintf(`DROP TABLE IF EXISTS %s`, pgIdentifier(p.tableName))); err != nil {
		return fmt.Errorf("drop reindex embeddings table: %w", err)
	}
	p.staging = false
	return nil
}

func (p *pgVectorStore) Close() error {
	if !p.ownsDB {
		return nil
	}
	return p.db.Close()
}

// pgVectorLiteral formats a []float32 as a pgvector literal string "[1.0,2.0,3.0]".
func pgVectorLiteral(v []float32) string {
	parts := make([]string, len(v))
	for i, f := range v {
		parts[i] = fmt.Sprintf("%g", f)
	}
	return "[" + strings.Join(parts, ",") + "]"
}
