# Multimodal RAG System for *The Batch*

**Project Description:** This project implements a **multimodal Retrieval-Augmented Generation (RAG)** system that lets users query a collection of AI news articles from [deeplearning.ai’s *The Batch*](https://www.deeplearning.ai/the-batch/) newsletter. It combines text and image information from the articles to answer user questions about AI topics. The system scrapes articles (and their images) from *The Batch*, indexes them for similarity search, and uses a large language model to generate answers with cited sources. The goal is to efficiently retrieve relevant articles and images for a query and have an AI (e.g. OpenAI’s GPT-4o) produce a comprehensive, source-grounded answer. This demonstrates how to integrate multiple data modalities (text and vision) into a RAG pipeline for a richer QA experience.

## System Architecture and Components

The system follows a pipeline architecture with distinct components for data ingestion, indexing, and query processing:

* **1. Data Scraper (`scraper.py`):** This module collects all *The Batch* articles and their images. It uses **Playwright** (headless Chromium browser) to navigate the pagination on deeplearning.ai’s search page and gather every article URL. This approach is necessary because The Batch’s archive is loaded dynamically – requiring clicking “Next page” to reveal older articles. Playwright can handle such dynamic, JavaScript-rendered content (with features like auto-waiting and browser automation) that static HTML scrapers would miss. Once URLs are collected, `scraper.py` uses **Requests + BeautifulSoup4** to fetch each article page concurrently and parse out the content:

  * It extracts the article **title**, **URL**, and **full text** (stripping headers or extraneous parts).
  * It finds the first relevant **image** in the article (if any) and downloads it (resolving any lazy-loaded URLs).
  * Each article is saved as a JSON file in `data/raw/articles` (with keys: `title`, `url`, `text`, `image_filename`), and a master JSONL file (`batch_articles.jsonl`) is appended for indexing.

* **2. Index Builder (`build_index.py`):** This stage creates vector indexes for efficient similarity search over the collected data. It uses **SentenceTransformers** models to encode the data:

  * A **text encoder** (`all-MiniLM-L6-v2`) converts each article’s text into a dense vector embedding.
  * An **image encoder** (`clip-ViT-B-32`, OpenAI’s CLIP model) converts each article’s image into an embedding. CLIP is a multimodal model that maps images and text into a shared vector space, making it powerful for matching images with relevant text descriptions.
  * Using these embeddings, the script builds **FAISS** indexes (one for text and one for images). FAISS (Facebook AI Similarity Search) is a library optimized for fast similarity search on dense vectors – even at large scale. We use a simple inner-product index (with normalized embeddings, this effectively means cosine similarity) for both modalities. Alongside the vectors, metadata (article title, URL, image filename) is stored in parallel Python lists.
  * Finally, it saves the indexes (`txt.index` for text, `img.index` for images) and metadata (`txt.meta.pkl`, `img.meta.pkl`) to the `data/index` directory. This indexing step is done **once** offline; the index files will be loaded by the app for all future queries, ensuring quick retrieval without re-computation.

* **3. Streamlit Application (`main.py`):** This is the user-facing interface and query engine. It’s a **Streamlit** web app that loads the pre-built indexes and allows users to search the knowledge base:

  * On startup, it loads the SentenceTransformer models and the FAISS indexes into memory (using `st.cache_resource` to avoid re-loading on each interaction).
  * The UI presents a text input for questions (e.g. *“What are recent advances in AI hardware?”*) and an optional image uploader. The user can also set a slider for the number of results (articles/images) to retrieve (default 5).
  * **Retrieval:** When the user submits a query, the app embeds the query text using the same text model, and searches the text index for the top-\$k\$ relevant articles. Simultaneously, it embeds the query with the CLIP model (treating the query text as if it were a description) to search for top-\$k\$ relevant images. This yields two result sets: a list of article hits and a list of image hits. These results are displayed in two tabs (“Articles” and “Images”) for the user’s reference – showing article titles (with links to the original URL) and thumbnails of images.
  * **Generation:** Next, the app constructs a prompt for the language model. It gathers the full text of the top articles (up to the token limit) as context, each preceded by a header containing its title and URL. Only the highest-ranked articles that fit within the model’s context window will be used. The user’s question is appended after the context. If the user uploaded an image with the query, it is attached to the prompt in the format expected by GPT-4o (as a base64-encoded image data URI in the `messages` payload). The system then invokes the **OpenAI GPT-4o** model (via the OpenAI API) to generate an answer. GPT-4o is chosen for its multimodal capability – it can accept both text and images in the prompt – and its large 128k token context window, which allows including multiple long articles as context. The prompt instructs the model to act as an expert assistant and **cite sources** in the answer. The model’s reply is finally displayed in an “Answer” section, with markdown links citing the relevant Batch articles. Each answer thus provides a synthesis of the retrieved information, grounded in The Batch’s content and accompanied by references.

## Key Design Choices and Rationale

* **Dynamic Page Scraping with Playwright:** The Batch’s website does not provide a simple list of all past articles; instead, content is revealed through a paginated search interface that loads results via JavaScript. We used Playwright to simulate a real browser, navigate through all “Next page” clicks, and capture every article link. This was a deliberate choice because traditional HTML scrapers (BeautifulSoup alone, etc.) cannot execute JavaScript and would miss the majority of content. Playwright’s ability to handle browser events ensured we got a complete dataset of articles.

* **SentenceTransformers for Text & Image Encoding:** For semantic search, we needed to convert both article text and images into comparable vector representations. We chose **`all-MiniLM-L6-v2`** for text due to its balance of speed and decent embedding quality for sentence-level semantics. For images, we chose **CLIP ViT-B/32** because CLIP is explicitly designed to bridge image and text domains. Using CLIP means a text query can be embedded in the same space as images, allowing the system to find images relevant to a textual query (e.g. a query about “robotics” might surface an image from an article about robots). Both models are lightweight enough to run on CPU, which aligns with our goal of a fast, local indexing process.

* **FAISS for Vector Similarity Search:** We use FAISS indexes to store and query embeddings. FAISS is an industry-standard library for fast nearest-neighbor search on dense vectors, capable of handling millions (even billions) of vectors efficiently. It allows our system to retrieve relevant articles/images in milliseconds, which is crucial for a responsive user experience. We opted for a simple **IndexFlatIP** (inner product) index with normalized vectors (equivalent to cosine similarity) given the moderate dataset size. This design choice provides simplicity and speed without needing complex index structures.

* **OpenAI GPT-4o as the Default Reasoning Engine:** We selected GPT-4o (GPT-4 “omni”) for answer generation due to its multimodal understanding and large context window. GPT-4o is a next-generation GPT model that can **accept both text and images as input**, and is more efficient and cost-effective than previous GPT-4 versions. Importantly, its 128k token context limit allows us to feed in several long articles at once without truncation. This means the model can reason over a substantial knowledge base in one go. GPT-4o’s multimodal capability is leveraged when users provide an image: the model can analyze the image (e.g. interpreting a chart or figure from an AI report) and incorporate that understanding into its answer. We use the OpenAI API’s `gpt-4o-mini` model for generation by default – it provides faster and cheaper responses (albeit with slightly lower quality than full GPT-4o) and is suitable for our use case. The prompt is tuned to encourage thorough answers with **markdown citations**, ensuring traceability of facts to the source articles.

* **Local LLM Fallback (via llama-cpp):** To make the system accessible even without OpenAI API access (or to avoid ongoing API costs), we designed it with a fallback option to use a local Large Language Model. In a non-API mode, one could run a model like **Llama 2** (in GGUF format) via the `llama-cpp-python` library. The repository provides a `data/models` directory intended for storing such a model. The idea is that if `OPENAI_API_KEY` is not set, the app could load a local model (e.g. a 13B parameter Llama 2 chat model) and use it to generate answers instead of GPT-4o. This design choice was made to ensure the system can run fully offline and free-of-charge, albeit with some trade-offs:

  * The local model’s responses may not be as accurate or coherent, and most open-source models currently cannot directly process images. Thus, the multimodal answering (image understanding) is only truly available with GPT-4o. The text retrieval and search would still work with a local model.
  * We consider the local mode a fallback for development or demos without API access. In production or for the best results, GPT-4o is recommended for its superior reasoning and multimodal abilities.

* **Preprocessing & Indexing Once (Offline) vs Real-Time:** A key design decision was to **precompute the embeddings and indexes in advance**. Scraping and embedding hundreds of articles (each possibly several hundred words long) is time-consuming. By doing this offline (via `scraper.py` and `build_index.py`), the Streamlit app (`main.py`) can start up quickly and handle queries in real-time with minimal overhead. The app only needs to embed the user’s query and perform fast FAISS lookups at runtime. This leads to a snappy user experience – the heavy lifting (data gathering and model encoding) is done only one time. In a live system, new articles from *The Batch* can periodically be added by re-running the scraper and updating the index, but query-time processing remains light.

## Setup and Installation

Follow these instructions to set up the project locally:

**Prerequisites:**

* **Python 3.12** (the project uses Python 3.12 — see the [.python-version](./.python-version) which specifies `3.12.11`). Using the matching version is recommended for compatibility.
* An OpenAI API key (if you plan to use GPT-4o for answer generation). You can obtain one from the OpenAI dashboard. *Note:* GPT-4o access may be limited to certain accounts; ensure your API key has access to the `gpt-4o` or `gpt-4v` model family.

**1. Clone the Repository:**

```bash
git clone https://github.com/YourUsername/batch-multimodal-rag.git
cd batch-multimodal-rag
```

**2. Create a Virtual Environment (optional but recommended):**

```bash
python3.12 -m venv venv
source venv/bin/activate  # on Linux/Mac
venv\Scripts\activate     # on Windows
```

**3. Install Dependencies:**
All required Python packages are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

This will install packages including:

* **Streamlit** (for the web UI),
* **faiss-cpu** (Faiss for vector search),
* **sentence-transformers** (for embedding models),
* **Pillow** (image processing),
* **Requests** (HTTP requests for scraping),
* **OpenAI** (OpenAI API client),
* **python-dotenv** (to load environment variables),
* **tiktoken** (tokenization for GPT-4 models),
* plus their sub-dependencies. Ensure these install without errors. *(If using a system like Ubuntu in Docker, the Dockerfile already handles system deps like `libgomp1` for Faiss. On a local machine, if you encounter Faiss installation issues, you might need to install a C++ build tool or use conda for Faiss.)*

**4. Set up Environment Variables:**
Create a file named `.env` in the project root (this is git-ignored) with the following content:

```ini
OPENAI_API_KEY=your-openai-api-key-here
```

This key is required for `main.py` to call the OpenAI GPT-4o API. If you plan to use a local model only, you can omit this or leave it blank, but then you should adjust the code to use llama-cpp (currently not automatic). You can also configure other environment settings here if needed (e.g., proxy settings or paths), but none others are strictly required by the code.

**5. Data Acquisition – Scraping The Batch:**
*If you already have the data files, you can skip this step.* Otherwise, run the scraper to fetch the articles and images:

```bash
python scraper.py
```

Before running, ensure Playwright is set up. You may need to install browser drivers by running `playwright install` (the scraper uses Chromium). The scraper will launch a headless browser (it is set to `HEADLESS=False` by default in the code – you can change it to `True` to hide the browser window). It will iterate through all pages of The Batch’s search results, collect article URLs, then download each article’s content. This process may take a while (several minutes, depending on network speed and number of articles).

Upon completion, you should have a `data/raw/articles` directory filled with JSON files (one per article) and a `data/raw/images` directory with image files. The console output will indicate how many articles were saved (for example, “✅ done — saved 150 articles” if 150 articles were found).

**6. Build the Vector Index:**
Next, create the Faiss indexes for fast retrieval:

```bash
python build_index.py
```

This will read all articles from `data/raw/batch_articles.jsonl`, embed the text and images, and write out the index files to `data/index/`. You should see a message at the end like “🏁 Index build complete: X articles | Y images” indicating the number of text and image embeddings indexed. After this step, the system is ready to serve queries.

**7. Run the Streamlit App:**
Launch the Streamlit interface with:

```bash
streamlit run main.py
```

By default, Streamlit will open a local web browser at `http://localhost:8501`. You should see the application titled "📰 Multimodal RAG for *The Batch*" with the query input ready.

*Note:* The first run of `main.py` will load the SentenceTransformer models. These will be downloaded from Hugging Face on first use (\~100 MB for the two models) and cached in `./data/models` (as we set `HUGGINGFACE_HUB_CACHE` to that path). This may take a minute. Subsequent runs will use the cached models.

**8. (Optional) Using Docker:**
For convenience or deployment, a Docker setup is provided:

* Ensure Docker (and docker-compose if needed) is installed on your system.
* Build the image using the provided **Dockerfile**:

  ```bash
  docker build -t batch-rag-app .
  ```

  This will create a container image with Python 3.12-slim, install the requirements, and copy in the `data/` folder and `main.py`.
* Run the container:

  ```bash
  docker run -p 8501:8501 -e OPENAI_API_KEY=your-openai-key-here batch-rag-app
  ```

  Then access `http://localhost:8501` as usual.

  Alternatively, you can use **Docker Compose** with the included `docker-compose.yml`. First, ensure you’ve set `OPENAI_API_KEY` in your environment or in the compose file. Then run:

  ```bash
  docker-compose up --build
  ```

  This will build and start the service on port 8501. The compose file simply automates the build and passes the API key into the container.

## Deployment Instructions

If you want to deploy this application for others to use, there are a couple of cloud-friendly approaches. The app is designed to run as a self-contained Docker container (no external database or services needed other than OpenAI API), which makes cloud deployment easier.

**Deploying on Google Cloud Run:**
Google Cloud Run is a serverless container platform that can run Docker images and automatically scale them. To deploy:

1. **Containerize**: Make sure you have built the Docker image (`batch-rag-app`) or have a Dockerfile ready. You can use Google Cloud Build to build the image, or build locally and push to a registry like Google Container Registry or Docker Hub.
2. **Deploy**: Use the gcloud CLI or Cloud Console. For example, with gcloud:

   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/batch-rag-app:v1 .
   gcloud run deploy batch-rag --image gcr.io/YOUR_PROJECT_ID/batch-rag-app:v1 --platform managed --memory 2Gi --cpu 1 --allow-unauthenticated --port 8501
   ```

   During deployment, set the environment variable `OPENAI_API_KEY` in the Cloud Run configuration (you can do this via `--set-env-vars` flag in the deploy command or in the Cloud Console UI). For instance: `--set-env-vars OPENAI_API_KEY=your-key`. Also, consider setting a concurrency limit (Streamlit can handle multiple users, but heavy LLM calls might warrant limiting to maybe `--concurrency 1` or 2 for reliability).
3. **Domain and Usage**: Cloud Run will give you a URL for the app. You can use it directly or map a custom domain. Cloud Run will scale the container down to zero when idle, meaning you won’t be charged when nobody is using it. This makes it cost-effective for demos or infrequent use. The Cloud Run **free tier** typically covers a significant amount of runtime (e.g., 2 vCPU hours per day, 1 GiB memory, and some networking) which might be enough for light usage. Just be mindful that usage of the OpenAI API will still incur costs as per OpenAI’s pricing.

**Deploying on AWS EC2:**
If you prefer a traditional VM approach:

1. Launch an EC2 instance (e.g., an Ubuntu 22.04 LTS server). A relatively small instance (2 vCPU, 4 GB RAM) should handle the app for a few users. You can use a free-tier eligible instance like t2.micro for testing, but it might be underpowered especially if you plan to run a local LLM model – consider t3.small or t3.medium for better performance.
2. Install Docker on the EC2 instance (Amazon Linux AMI may have Docker pre-installed, or use `sudo apt-get install docker.io` on Ubuntu).
3. Transfer your code or use git to clone the repo on the server.
4. Build and run the Docker container on the EC2 similarly to local instructions:

   ```bash
   docker build -t batch-rag-app .
   docker run -d -p 8501:8501 -e OPENAI_API_KEY=... batch-rag-app
   ```

   Ensure the EC2 security group allows inbound traffic on port 8501 (or map it to 80 via a proxy if preferred).
5. Access the app via the EC2’s public IP or public DNS on the specified port. You may want to set up a more permanent domain or use AWS Elastic IP for convenience.
6. **Cost considerations:** EC2 does not auto-scale to zero, so you will be charged for the instance as long as it’s running. Check AWS pricing for your instance type; for example, a t3.small might cost around \$0.02 per hour (\~\$14/month). There’s no built-in free tier indefinitely (beyond the first-year free usage for new accounts). If cost is a concern, ensure you shut down or pause the instance when not in use, or use Cloud Run instead. Also factor in the OpenAI API cost for usage – GPT-4o-mini is cheaper than GPT-4, but it still costs money per prompt (at the time of writing, GPT-4o API was roughly \$5 per million input tokens and \$15 per million output tokens, which is half the cost of GPT-4 Turbo).

**Note:** When deploying publicly, consider securing the app (Cloud Run can require authentication, or you might put the app behind a simple login) especially since it uses an API key. You wouldn’t want someone else to abuse your OpenAI credits. Also monitor the OpenAI usage dashboard for any spikes and set limits if needed.

## Usage Guide

Once the app is running (locally or in the cloud), using it is straightforward and user-friendly:

1. **Open the App**: Navigate to the web interface (e.g., `http://localhost:8501` or your deployed URL). You’ll see the title *“Multimodal RAG for The Batch”* and a brief note about which model is in use (by default, it will say “You are using GPT-4o”).

2. **Ask a Question**: Enter your query in the text box. This could be any question or topic related to AI news covered by *The Batch*. For example:

   * *“What did *The Batch* report about self-driving cars in 2023?”*
   * *“Summarize the latest trends in AI chips.”*
   * *“What is Andrew Ng’s stance on AI regulation, according to The Batch?”*

   The system works best for topics that are likely discussed in the *The Batch* articles (which span various AI research, business, and societal topics). It performs a semantic search, so you can use natural language questions.

3. **(Optional) Upload an Image**: If you have an image that relates to your query, you can upload it using the **“Optional image”** uploader. This step is truly optional – it’s there to showcase GPT-4o’s multimodal capability. For instance, you might upload a chart or graph from an AI report, or a photo from an AI experiment, and ask a question about it. The app will include this image in the prompt to GPT-4o. *Keep in mind:* The image will not be used to find articles (it doesn’t perform reverse image search); instead, the image is only analyzed by the LLM to enrich the answer. So the use-case might be asking something like “What does this chart indicate about AI model performance? Have there been related discussions in *The Batch*?” where GPT-4o can interpret the chart and combine that with knowledge from the articles. **Note:** Image understanding in answers only works with GPT-4o. If you’re using a local model fallback, the image will be ignored by that model (it can’t process images).

4. **Configure Retrieval (optional)**: There is a slider labeled **“Max articles/images”** which defaults to 5. This controls how many text articles and how many images will be retrieved at most. You can increase this up to 10 for a broader search, or lower it for a more narrow search. The top results are determined by vector similarity to your query.

5. **Submit the Query**: Click the **“Search”** button. The app will show a spinner while it:

   * Embeds your query and searches the indexes.
   * Displays the results in the **Articles** and **Images** tabs. In the “Articles” tab, you’ll see a list of the top articles: each entry shows the article’s title (clickable to the original URL) and, if available, a thumbnail of the article’s image. In the “Images” tab, you’ll see thumbnails of images that were deemed relevant, each with a caption of the article title (also linked). This lets you quickly verify what the retrieval step found.
   * The retrieved articles’ content is then assembled into a prompt (with as many as can fit in GPT-4o’s input limit, prioritized by relevance).

6. **View the Answer**: After the retrieval step, the app will show another spinner indicating it’s “Generating answer…”. GPT-4o (or the local model) is now reading the context and formulating a response. Once done, the **Answer** section will appear. This will typically be a few paragraphs of analysis or summary that addresses your question. Critically, the answer will include **citations** in markdown format – for example, *“… as noted in a recent report.”* Each citation corresponds to one of *The Batch* articles used, and is linked. You can click the citation to open the source article. The model was instructed to cite *each* article it used at least once, so you should see references sprinkled throughout the answer. This helps you trust but verify the content. The combination of retrieval + generation means the answer is grounded in actual articles (reducing hallucination and increasing relevance).

7. **Switching the LLM (OpenAI vs Local)**: In the current UI, there isn’t a toggle to switch models on the fly – it defaults to using OpenAI GPT-4o when an API key is present. However, you can run the system without an API key by adjusting the code: for example, instead of calling OpenAI, you could load a local model with `llama-cpp-python`. (This would involve using a command like `Llama(model_path=..., n_ctx=...)` and feeding it the prompt). In a future update, a dropdown could be added to select “OpenAI GPT-4o” or “Local LLM” at runtime. For now, if you want to use the local model, you’ll need to modify `main.py` accordingly. Keep in mind the limitations mentioned: a local model likely won’t handle images and may require a reduced context size (e.g., 2048 or 4096 tokens). Ensure your local model is placed in `data/models` and update the code to load it. The UI would remain the same; only the answer generation backend changes. When using the local model, the note under the title can be changed from “You are using GPT-4o” to something like “You are using Local LLM” to reflect that.

8. **Examples**: Try queries to explore the system’s capabilities. For instance:

   * *“Give me a summary of the AI highlights from The Batch in October 2024.”* – The system might retrieve articles from that period and compile a summary with citations.
   * *“What does The Batch say about AI and climate change?”* – It will find any articles touching on climate applications of AI.
   * *Upload an image* (say, a picture of a humanoid robot) and ask: *“How does this relate to recent AI developments?”* – GPT-4o might identify the robot in the image and recall related news from The Batch about robotics.
   * Explore the image search tab by entering a topic – e.g., *“quantum computing”* – and then clicking the **Images** tab to see what visuals The Batch has published for that topic.

9. **Performance and Limits**: The system is intended for interactive use. Each query will typically take a few seconds for retrieval and then the GPT model’s response time (which could be 5-15 seconds for GPT-4o-mini on a short answer, longer for very detailed answers that use many tokens). The context assembly tries to stay within \~60k tokens for GPT-4o. If you select a very high number of articles and they are all long, the app will include as many as fit and drop the rest. It starts with a 500-token allowance for the first article and \~180 tokens for each additional (this heuristic is adjustable via `ARTICLES_PER_TOKEN_BUDGET`). Practically, 5 articles of a few paragraphs each will fit easily in GPT-4o’s input. If you somehow retrieve near the limit, GPT-4o-mini might shorten the answer to stay within output limits.

## Project Structure and Data

The repository is organized with a clear separation of data and code. Key directories under the project root include:

```
📂 data/
   📂 raw/
       📂 articles/        # Scraped articles in JSON format (one file per article)
       📂 images/          # Downloaded images from articles (jpg, png, etc.)
       📄 batch_articles.jsonl  # All articles in one JSONL (each line is a JSON record)
   📂 index/
       📄 txt.index        # FAISS index for text embeddings
       📄 img.index        # FAISS index for image embeddings
       📄 txt.meta.pkl     # Python pickle of metadata for text index entries
       📄 img.meta.pkl     # Python pickle of metadata for image index entries
   📂 models/
       (cache and model files; see below)
📂 src/ (if code structured in a module, otherwise files in root)
   📄 scraper.py
   📄 build_index.py
   📄 main.py
   📄 Dockerfile
   📄 docker-compose.yml
   ... (etc)
```

* **`data/raw/articles/`**: Contains JSON files for each article scraped. The filename typically is based on the article’s slug (e.g., an article at URL `https://www.deeplearning.ai/the-batch/my-news-title/` becomes `my-news-title.json`). Each JSON has the structure:

  ```json
  {
    "title": "...",
    "url": "https://www.deeplearning.ai/...",
    "text": "...",              // full text content of the article
    "image_filename": "..."     // filename of the downloaded image (if any)
  }
  ```

  The text is cleaned to remove any boilerplate (like “Article” labels) and contains the main content (paragraphs and list items joined).

* **`data/raw/images/`**: Stores image files for articles. If an article has an image, the scraper saves the first image it finds. The `image_filename` in the article JSON corresponds to a file in this folder. For example, if `image_filename` is `my-news-title.jpg`, the file will be `data/raw/images/my-news-title.jpg`. These images are used for two purposes: (a) generating the image embeddings for the FAISS image index, and (b) displaying thumbnails in the Streamlit UI results. *Note:* Not every article has an image; if `image_filename` is `null` or empty, that article just won’t have a thumbnail or image vector.

* **`data/index/`**: Contains the serialized FAISS indexes and accompanying metadata:

  * `txt.index` – the binary index file storing all text article embeddings. We used a flat index, so this file essentially has the raw vectors.
  * `img.index` – the index of image embeddings.
  * `txt.meta.pkl` – a pickle file (Python list) of metadata dictionaries for text embeddings. Each entry corresponds to one vector in `txt.index` and contains fields like `slug`, `title`, `url`, `img` (filename).
  * `img.meta.pkl` – metadata for image vectors. Entries contain `slug`, `img` (filename), `title`, `url`. The slug can be used to map an image back to its article (and indeed many image entries will have a matching text entry with the same slug).
    These metadata files allow the app to retrieve the context (full text) and display titles without needing to store that in the FAISS index itself. **Note:** The combination of FAISS index and pickle meta is a simple solution. In a larger system, one might use a vector database or at least keep IDs to fetch metadata on the fly. Here we opted for simplicity – loading everything into memory (which is feasible since the dataset is not huge, on the order of hundreds of articles).

* **`data/models/`**: This directory is used for models:

  * During indexing and query, we set Hugging Face cache to this folder, so the SentenceTransformer model files (for MiniLM and CLIP) will be stored here. This prevents them from polluting the default cache (and makes it easy to include them if packaging the app in Docker).
  * If using a local LLM, you would place the model weights here as well. For instance, you might download a `llama-2-7b-chat.gguf` or similar and put it in `data/models`. That way it’s ignored by git (see `.gitignore`) but available to load in the app. In our current implementation, we don’t automatically load it, but you can modify `main.py` to do so if needed.

* **Code files**: Apart from the three main Python scripts discussed, the repo includes:

  * `Dockerfile` – defines the container build (uses Python 3.12 slim, installs deps, copies `data` and `main.py`, and sets the entrypoint to run Streamlit).
  * `docker-compose.yml` – a simple compose file to build/run the container with an environment variable for the API key.
  * `.env.example` (if provided) – sometimes projects include a template; otherwise, we described `.env` above.
  * `README.md` – (this file).
  * Possibly other support files or notebooks (not mentioned in prompt, but for completeness, e.g., an archive folder or tests if any).

## License and Attribution

**License:** The source code in this project is released under the MIT License (an open-source license that permits reuse with attribution). You are free to use, modify, and distribute the code in this repository as per the terms of the MIT License. (If this project is part of an assignment or specific program, please adhere to any submission or usage guidelines provided by the organizers.)

**Data Attribution:** The articles and images used by this project are sourced from [**deeplearning.ai**](https://deeplearning.ai)’s publication *The Batch*. All content from *The Batch* (text and images) is **©** deeplearning.ai and its authors. This project is for educational and research purposes – demonstrating a RAG application – and is **not affiliated with or endorsed by deeplearning.ai**. When using or deploying this system, **do not commercialize or widely distribute the scraped content** without permission from the content owners. If you present results from this app (e.g., in a blog or demo), consider crediting "*The Batch (deeplearning.ai)*" as the source of the articles.

**Model and Library Attributions:**

* The embedding models come from the **SentenceTransformers** library. Specifically, *“all-MiniLM-L6-v2”* was developed by SBERT/Google and *“clip-ViT-B-32”* is based on OpenAI’s CLIP. These models are distributed under permissive licenses (Apache/MIT) via Hugging Face. We thank the authors for providing these powerful embeddings out-of-the-box.
* **OpenAI GPT-4o** (and GPT-4 family) is a proprietary model by OpenAI. Usage of the OpenAI API is subject to OpenAI’s terms and policies. Ensure you follow those terms, especially regarding user data and output usage. This project’s use of GPT-4o is compliant with their guidelines (it’s used to generate answers with attribution).
* **Playwright** is a Microsoft OSS project (Apache 2.0 License) – thanks to its maintainers for a great tool to handle web automation.
* **FAISS** is by Facebook AI Research (licensed under MIT) – credit to the FAISS team for enabling fast vector search.
* Other Python libraries like Streamlit, etc., each come with their respective licenses (Streamlit is Apache 2.0). We acknowledge these open-source tools that make this project possible.

**Acknowledgements:** This project was developed as part of a learning program (Gen AI Lab onboarding test task) to practice building multimodal RAG systems. We appreciate the guidance and recommendations (such as using GPT-4o, Claude, or similar) from the program. The design and implementation choices were made to create a functional system that integrates text and vision in a user-friendly way.

Finally, if you use or build upon this project, please maintain this README’s attributions and cite the original sources (*The Batch* articles, etc.) as we have done. Happy researching with The Batch!
