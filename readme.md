# Pix - Visual Retrieval-Augmented Generation

## Overview

Pix is an experimental implementation of a visual Retrieval-Augmented Generation (RAG) pipeline for document understanding, with a particular focus on technical diagrams and visually structured information.

Traditional text-based RAG systems typically extract text from documents, divide it into chunks, embed those chunks, and retrieve relevant passages in response to a query. However, this approach can lose information encoded in spatial arrangement, visual grouping, connectors, and relationships between components.

Pix explores an alternative approach: **retrieve relevant visual regions of a document using multimodal embeddings, then use a vision-language model to interpret those regions.**

The primary implementation is contained in [`pix_RAG.ipynb`](pix_RAG.ipynb). The notebook uses [PixelRAG](https://github.com/StarTrail-org/PixelRAG) for document indexing and retrieval, followed by a vision-language model for detailed analysis of retrieved image tiles.

The objective is to investigate whether this retrieval-and-analysis architecture can support questions that require understanding not only what a diagram contains, but also how its components are organized and connected.

## Core Implementation

The notebook implements a two-stage pipeline: **visual retrieval followed by vision-language analysis**.

### 1. Document Ingestion and Configuration

The notebook accepts PDF documents and image files (`.png`, `.jpg`, and `.jpeg`) through the Google Colab upload interface.

Uploaded files are placed in `my_docs/`, which serves as the input corpus for indexing.

The notebook then generates a `pixelrag.yaml` configuration specifying:
- **Embedding model:** `Qwen/Qwen3-VL-Embedding-2B`
- **Execution device:** CUDA, using GPU 0
- **Embedding backend:** Direct GPU execution
- **Tile dimensions:** 512 × 512 pixels
- **Tile overlap:** 256 pixels
- **Index output directory:** `my_index/`

The overlapping tiling configuration allows adjacent regions to share visual context. This is useful when labels, connectors, or other diagram elements extend across tile boundaries.

### 2. Visual Indexing and Embedding

Pix uses PixelRAG to build a searchable index from the uploaded documents.

The indexing process generates visual tiles and associated metadata, including the information needed to map retrieval results back to their corresponding document regions.

The resulting index is stored in `my_index/`. The notebook checks whether an index already exists and skips rebuilding it unless explicitly instructed to do so.

This separates the relatively expensive indexing stage from subsequent query execution.

### 3. Retrieval Through PixelRAG

Once indexing is complete, the notebook launches the PixelRAG serving process against the generated index.

The server exposes a local HTTP endpoint at `127.0.0.1:30001`, which the notebook uses to submit natural-language queries.

Each search request specifies the query and the number of results to retrieve. The response contains ranked hits with metadata such as:
- Retrieval score
- Article identifier
- Tile index
- Chunk index
- Source path

The notebook supports both predefined questions and interactive query input.

The predefined questions emphasize technical diagram understanding, including component hierarchies, connections across system boundaries, data-flow paths, dependencies, and potential cycles in directed relationships.

At this stage, the system identifies relevant visual regions; it has not yet performed the notebook's subsequent, detailed vision-language analysis.

### 4. Vision-Language Analysis of Retrieved Regions

Retrieval identifies candidate regions, but a similarity score alone does not provide a detailed explanation of the visual content.

Pix therefore applies a separate vision-language model to selected retrieval results.

The configured model is `meta-llama/Llama-3.2-11B-Vision-Instruct`, loaded through Hugging Face Transformers and configured for 4-bit NF4 quantization using BitsAndBytes.

This stage performs the following operations:

1. **Resolve the retrieved tile:** Use the article identifier and chunk metadata to locate the corresponding tile and its manifest under `my_index/tiles/`.
2. **Load the image:** Read the exact visual region returned by retrieval.
3. **Construct the prompt:** Supply the image and a structured technical-diagram extraction prompt to the vision-language model.
4. **Generate an interpretation:** Extract visible entities, hierarchy, labels, and relationships while explicitly representing uncertainty.
5. **Associate the result with retrieval metadata:** Preserve the relationship between the generated description, its source tile, and its retrieval score.

The extraction prompt emphasizes evidence-based interpretation. It instructs the model to preserve visible labels, avoid inventing components or connections, distinguish actual connectors from mere spatial proximity, and mark ambiguous or unreadable information rather than guessing.

The intended output is organized into four sections:

- **Overview:** Diagram type, layout, dominant flow direction, and extraction status.
- **Entities:** Distinct visual elements, their labels, parent relationships, types, metadata, and positions.
- **Relationships:** Connections supported by visible connectors and their endpoints.
- **Uncertainty:** Partial, ambiguous, or unreadable information that cannot be reliably established from the image.

This structured approach is intended to make the model's interpretation more systematic and easier to inspect than an unrestricted image caption.

### 5. Result Inspection and Export

The final stage combines retrieval metadata with generated visual interpretations.

For the selected query, the notebook:
- Retrieves the requested number of candidate regions.
- Resolves the corresponding tile images.
- Generates a description for each selected region.
- Displays the tile alongside its retrieval score and generated description.
- Records execution failures and continues processing other hits where possible.
- Saves the generated descriptions to `captions.md`.

The Markdown output associates each result with its rank, tile index, retrieval score, and generated text.

The notebook also provides a download step so that the resulting report can be retrieved from Google Colab.

The current final execution cell demonstrates the workflow using the first query in the active query list and retrieves one tile for inspection.

## Architecture

The overall workflow can be represented as follows:

```text
                 Input Documents
                PDF / PNG / JPG
                       |
                       v
               Document Ingestion
                  my_docs/
                       |
                       v
              PixelRAG Index Build
                       |
             +---------+---------+
             |                   |
             v                   v
      Visual Tiles          Embeddings
             |                   |
             +---------+---------+
                       |
                       v
                 Search Index
                  my_index/
                       |
                       v
                Natural-Language
                     Query
                       |
                       v
               PixelRAG Search API
                 localhost:30001
                       |
                       v
              Ranked Retrieval Hits
                       |
                       v
              Resolve Relevant Tiles
                       |
                       v
             Vision-Language Model
             Llama 3.2 11B Vision
                       |
                       v
             Structured Interpretation
                       |
                       v
              Visual Inspection
               and Markdown Export
```

The key architectural distinction is that retrieval and interpretation are separate operations.

PixelRAG identifies candidate regions using the embedding-based search pipeline. The vision-language model then examines the selected images to produce a more detailed semantic interpretation.

This separation allows retrieval parameters and visual-analysis behavior to be examined independently.

## Technology Stack

| Component | Technology | Role |
|---|---|---|
| Retrieval and indexing | [PixelRAG](https://github.com/StarTrail-org/PixelRAG) | Document processing, visual indexing, and search |
| Embedding model | [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) | Multimodal representations for retrieval |
| Vision-language model | [Llama 3.2 11B Vision Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) | Interpretation of retrieved visual regions |
| Model quantization | [BitsAndBytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | 4-bit NF4 model loading |
| Deep learning framework | [PyTorch](https://pytorch.org/) | GPU-backed model execution |
| Model interface | [Transformers](https://huggingface.co/docs/transformers/index) | Model and processor loading and inference |
| Dependency management | [uv](https://docs.astral.sh/uv/) | PixelRAG environment synchronization |
| Execution environment | Google Colab | Notebook execution and GPU access |

## Repository and Runtime Artifacts

The notebook generates or uses the following files and directories:

| Path | Purpose |
|---|---|
| `pix_RAG.ipynb` | Main notebook containing the end-to-end workflow |
| `my_docs/` | Uploaded source documents |
| `pixelrag.yaml` | Indexing and embedding configuration |
| `my_index/` | Generated index, tile images, manifests, and associated metadata |
| `pixelrag_run.log` | Execution log containing stage timings, diagnostics, and error tracebacks |
| `captions.md` | Markdown report containing generated descriptions of retrieved regions |

The PixelRAG repository and its Python environment are created under `/content/PixelRAG` in the Colab runtime.

These artifacts are runtime outputs; they are not necessarily committed to the Pix repository.

## Running the Notebook

### Prerequisites

- A Google account with access to Google Colab.
- A GPU-enabled runtime with sufficient available VRAM for both the embedding service and the vision-language model.
- A Hugging Face account with access to the selected Meta Llama checkpoint.
- A Hugging Face access token configured as `HF_TOKEN` in Colab Secrets.
- PDF documents or supported image files for indexing.

### Execution Steps

1. Open `pix_RAG.ipynb` in Google Colab.
2. Select a GPU runtime.
3. Configure `HF_TOKEN` in Colab Secrets and verify access to the model repository.
4. Execute the setup cells to check GPU availability, clone PixelRAG, and synchronize its dependencies.
5. Upload the documents to be indexed.
6. Generate `pixelrag.yaml` and build the visual index.
7. Start the PixelRAG server and verify that it remains running.
8. Enter custom queries or use the predefined technical-diagram questions.
9. Load the vision-language model and execute the retrieval-and-analysis stage.
10. Inspect the generated descriptions and download `captions.md`.

Run the cells in sequence because later stages depend on the index, running search server, and loaded vision-language model created earlier.

### GPU Memory Considerations

The notebook uses the GPU for embedding-based retrieval and loads a separate vision-language model for image interpretation.

Although the vision-language model is configured for 4-bit quantization, quantization does not eliminate GPU memory requirements. Model weights, runtime allocations, image processing, and inference can still require substantial VRAM.

The notebook includes GPU-memory diagnostics and checks for common failures. If the retrieval server continues to occupy GPU memory while the vision-language model is loaded, the available memory may be insufficient for both workloads.

The notebook should therefore be run on a runtime whose available memory has been verified for the selected configuration.

## Scope and Limitations

Pix is an experimental implementation rather than a production-ready document-understanding service.

Its current scope is centered on retrieving and interpreting selected visual regions. Several limitations follow from this design:

- **Retrieval coverage:** The vision-language model only analyzes the regions selected by retrieval. Relevant information omitted from the retrieved tiles may not appear in the final interpretation.
- **Cross-region reasoning:** Questions involving connections that span multiple tiles may require retrieving and examining additional regions. A single tile may not contain enough context to establish the complete relationship.
- **Spatial and structural accuracy:** The extraction prompt encourages evidence-based descriptions, but it does not guarantee that every label, connector, or endpoint will be interpreted correctly.
- **Generated output:** Model-generated descriptions may contain omissions or incorrect interpretations. Important relationships should be checked against the original document.
- **Resource constraints:** Indexing, embedding, and vision-language inference depend on available GPU memory, model access, and compatible software dependencies.
- **Operational scope:** The notebook demonstrates an end-to-end research workflow in Colab; it does not establish production-level reliability, scalability, or accuracy.

## References

- [Pix Repository](https://github.com/Arynshr/Pix)
- [PixelRAG](https://github.com/StarTrail-org/PixelRAG)
- [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)
- [Llama 3.2 11B Vision Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)
