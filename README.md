# Mini Support Agent for E-commerce

This project implements an AI support agent that combines Retrieval-Augmented Generation (RAG) for policy-related questions and structured data lookups for order-specific inquiries.

## Features

- **Intelligent Routing**: Uses Groq LLM to route queries between knowledge retrieval, order data lookup, or a combination of both.
- **RAG System**: Efficient semantic search over policy documents using `sentence-transformers`.
- **Similarity Search**: Cosine Similarity
- **Order Tracking**: Deterministic lookup for order status and details from CSV data.
- **Complex Reasoning**: Handles "chaining" queries, such as checking return eligibility based on order date and category-specific policies.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Variables**:
    Create a `.env` file from the example:
    ```bash
    cp .env.example .env
    ```
    Add your `GROQ_API_KEY` to the `.env` file.

3.  **Run the Agent**:
    ```bash
    python agent.py
    ```

4.  **Run Tests**:
    ```bash
    python run_tests.py
    ```

## Design Decisions

### Chunking & Retrieval Strategy
- **Chunking**: Documents are split into sections based on double newlines (`\n\n`). This preserves the context of paragraphs and headers while keeping chunks small enough for relevant retrieval.
- **Vector Store**: A simple in-memory vector store is used using `numpy` and `sentence-transformers` (`all-MiniLM-L6-v2`). Given the small size of the policy documents, a full-blown vector database (like Pinecone or Weaviate) was unnecessary. Semantic search is performed using cosine similarity.

### Routing Logic
The agent uses a two-step "reasoning" process:
1.  **Router**: A Groq-powered LLM analyzes the query and classifies it into `knowledge`, `order_data`, or `both`. It also extracts the `order_id` if present.
2.  **Execution**:
    - `knowledge`: Performs semantic search and generates an answer based on the retrieved context.
    - `order_data`: Fetches order details from the CSV and presents them.
    - `both`: First fetches order details, then retrieves relevant policy context, and finally passes both to the LLM to reason about the user's specific situation (e.g., return eligibility).

### Model Choice
- **LLM**: `llama-3.3-70b-versatile` via Groq. This model was chosen for its high performance, large context window, and low latency provided by the Groq inference engine. It handles complex reasoning and JSON routing tasks very reliably.
- **Embeddings**: `all-MiniLM-L6-v2` via `sentence-transformers`. It strikes a great balance between speed and accuracy for small-scale document retrieval.

## Test Cases
The `test_cases.json` file contains 8 representative questions covering:
- Pure knowledge questions (shipping, international shipping).
- Pure data questions (order status).
- Invalid order ID handling.
- Chained reasoning (return eligibility based on category and delivery date).

Detailed results can be found in `test_results.json`.
