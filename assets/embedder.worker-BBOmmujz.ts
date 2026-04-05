/// <reference lib="webworker" />
//
// Embedding Web Worker — runs entirely off the main thread.
//
// Message protocol
// ────────────────
// Incoming (main → worker):
//   { type: 'embed', corpusId, documents: { id, text, name }[] }
//
// Outgoing (worker → main):
//   { type: 'model-ready' }                                   model loaded & cached
//   { type: 'progress', corpusId, processed, total, docName } one doc embedded
//   { type: 'complete', corpusId, positions, model }          UMAP done
//   { type: 'error',   corpusId, message }                    unrecoverable failure

import { pipeline, env } from '@huggingface/transformers'
import { reduceToUMAP } from '../utils/corpus/umap'
import type { EmbeddingPoint } from '../types'

// ── Transformers.js config ────────────────────────────────────────────────────

// Fetch model files from the HuggingFace hub; cache in browser IndexedDB.
env.allowLocalModels = false
env.useBrowserCache  = true

// ── Constants ─────────────────────────────────────────────────────────────────

const MODEL_ID = 'Xenova/all-MiniLM-L6-v2'

// all-MiniLM-L6-v2 truncates at 256 tokens (~1,500 chars).
// We send a generous 8 k-char window so the model always sees the
// opening of each document — the most information-dense region.
const MAX_CHARS = 8_000

// ── Pipeline singleton ────────────────────────────────────────────────────────

type Extractor = Awaited<ReturnType<typeof pipeline>>
let extractor: Extractor | null = null

async function getExtractor(): Promise<Extractor> {
  if (!extractor) {
    extractor = await pipeline('feature-extraction', MODEL_ID, {
      progress_callback: (p: { status: string; progress?: number }) => {
        // Send download progress (0–100) while the model files are being fetched.
        // 'progress' status fires repeatedly during download; 'ready' means done.
        if (p.status === 'progress' && typeof p.progress === 'number') {
          self.postMessage({ type: 'model-progress', pct: Math.round(p.progress) })
        }
      },
    })
    self.postMessage({ type: 'model-ready' })
  }
  return extractor
}

// ── Message types ─────────────────────────────────────────────────────────────

interface EmbedRequest {
  type: 'embed'
  corpusId: string
  documents: { id: string; text: string; name: string }[]
}

// ── Main handler ──────────────────────────────────────────────────────────────

self.onmessage = async (e: MessageEvent<EmbedRequest>) => {
  const { corpusId, documents } = e.data

  try {
    const ext = await getExtractor()
    const rawPoints: { id: string; vector: number[] }[] = []

    for (let i = 0; i < documents.length; i++) {
      const { id, text, name } = documents[i]

      // Feature extraction — mean-pooled, L2-normalised 384-D vector.
      // The cast is safe: this pipeline always returns a rank-2 Tensor.
      const output = await (ext as (
        input: string,
        options: { pooling: string; normalize: boolean },
      ) => Promise<{ data: Float32Array }>)(text.slice(0, MAX_CHARS), {
        pooling: 'mean',
        normalize: true,
      })

      rawPoints.push({ id, vector: Array.from(output.data) })

      self.postMessage({
        type: 'progress',
        corpusId,
        processed: i + 1,
        total:     documents.length,
        docName:   name,
      })
    }

    // Dimensionality reduction: 384-D → 3-D
    const positions: EmbeddingPoint[] = reduceToUMAP(rawPoints)

    self.postMessage({ type: 'complete', corpusId, positions, model: MODEL_ID })
  } catch (err) {
    self.postMessage({
      type:    'error',
      corpusId,
      message: err instanceof Error ? err.message : String(err),
    })
  }
}
