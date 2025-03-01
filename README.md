# RAG System

This project implements a Retrieval-Augmented Generation (RAG) system using various libraries and tools. The system retrieves relevant documents based on a query and generates a response using a language model.

## Prerequisites

- Node.js
- npm
- Docker (for running the Ollama embeddings service)

## Setup

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd rag-system
    ```

2. Install dependencies:
    ```sh
    npm install
    ```

3. Create a `.env` file in the root directory and add the necessary environment variables:
    ```env
    # .env
    # Add any required environment variables here
    ```

4. Start the Ollama embeddings service:
    ```sh
    docker run -p 11434:11434 ollama/embeddings-service:latest
    ```

## Usage

1. Run the main script:
    ```sh
    npx tsx main.ts
    ```

2. The script will output the answer to the console based on the input question.

## Project Structure

- `main.ts`: The main script that initializes the RAG system, loads documents, splits them into chunks, indexes them, and defines the state graph for retrieval and generation.
- `README.md`: This file.

## Dependencies

- `cheerio`: For loading and parsing HTML documents.
- `dotenv`: For loading environment variables from a `.env` file.
- `@langchain/ollama`: For using the Ollama language model and embeddings.
- `langchain/vectorstores/memory`: For storing document embeddings in memory.
- `@langchain/community/document_loaders/web/cheerio`: For loading documents from the web using Cheerio.
- `@langchain/core/documents`: For handling document objects.
- `@langchain/core/prompts`: For handling prompt templates.
- `langchain/hub`: For pulling prompt templates from the LangChain hub.
- `@langchain/langgraph`: For defining and managing state graphs.
- `@langchain/textsplitters`: For splitting documents into chunks.

## License

This project is licensed under the MIT License.