# AI Toolkit

AI toolkit aims to be a simple easy-to-use pluggable library that can help your AI pipeline come out ahead.

The project does not aim to replace existing frameworks like `langchain` or `llama-index` but instead, aims to serve as a tool where you can combine and utilize the best from each framework.

## Packages

The library is divided into simple packages, with each package solving a challenge I faced when working on enterprise AI projects.

### `ai_toolkit.transforms`

`ai_toolkit.transforms` aims to be a simple document transformation pipeline with full async-support. It provides an easy interface for writing your own transformation, along with a pipeline and (optional) registry that can help orchestrate and manage these transformations.

The idea stems from data engineering, where we use a pipeline to transform our data. Each step in our pipeline takes in the set of data, modifies the data, and passes on the data to the next step.

This module aims to do the same with your documents. The entire system is written in a framework-agnostic way where a `Document` is just a collection of some text and metadata. Each transform takes in a sequence of `Document`s and returns back a sequence after processing. The pipeline orchestrates between these transformations, taking care of batching, error-handling, and logging.

Special credits to `llama-index`'s ingestion pipeline design, which was used as an inspiration for this design.

### `ai_toolkit.ocr`

`ai_toolkit.ocr` aims to allow combining different OCR providers for what works best for your data. It provides an easy base interface that (at its core) simply converts `File`s into `Document`s. This simple approach makes it really easy to plug it into your existing OCR pipeline and simplify implementation.

## Integrations

### `langchain`

`ai_toolkit.langchain` provides easy utilities to integrate `ai_toolkit` with the `langchain` AI framework.

### `llamaindex`

`ai_toolkit.llama_index` provides `liteparse` as an easy to use OCR module.

### `azure`

`ai_toolkit.azure` provides easy integratio with various Azure AI services such as Azure Document Intelligence.

## Upcoming

I have a lot planned for this toolkit. Some items in my pipeline are:

- Chunking - To be made available as a `DocumentTransform`. Easy plugins for trying out different chunking methods from your favourite framework.
- Index management - Simple easy-to-use index management that works across different vector stores.
- Prompt registry - Swappable prompt registry system that lets you simplify prompt management.
- Config managemnt - Tying up everything we worked on, a simple config management system with support for external config files such as .toml files.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
