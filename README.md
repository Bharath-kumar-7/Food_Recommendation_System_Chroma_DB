<div align="center">

# 🍽️ Food Recommendation System with ChromaDB

### An AI-powered food recommendation engine combining semantic vector search with IBM watsonx.ai's Granite LLM for intelligent, conversational food discovery.

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20DB-FF6B35?style=for-the-badge)](https://www.trychroma.com/)
[![IBM watsonx.ai](https://img.shields.io/badge/IBM%20watsonx.ai-Granite%20LLM-054ADA?style=for-the-badge&logo=ibm&logoColor=white)](https://www.ibm.com/watsonx)
[![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-Embeddings-FF9900?style=for-the-badge)](https://www.sbert.net/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Dataset](#-dataset)
- [Usage](#-usage)
  - [Interactive Search](#1-interactive-search)
  - [Advanced Filtered Search](#2-advanced-filtered-search)
  - [Calorie Checker](#3-calorie-checker)
  - [Enhanced RAG Chatbot](#4-enhanced-rag-chatbot)
  - [System Comparison](#5-system-comparison)
- [How It Works](#-how-it-works)
- [Module Reference](#-module-reference)
- [Example Outputs](#-example-outputs)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌟 Overview

The **Food Recommendation System** is an intelligent, multi-modal food discovery platform that leverages cutting-edge technologies:

- **ChromaDB** as a high-performance vector database to store and retrieve food embeddings
- **Sentence Transformers** (`all-MiniLM-L6-v2`) for generating rich semantic embeddings from food descriptions
- **IBM watsonx.ai Granite 3.3 8B Instruct** for generating natural language, context-aware food recommendations through a RAG (Retrieval-Augmented Generation) pipeline

Users can search for foods using natural language queries — such as *"spicy comfort food for a cold evening"* or *"light Italian dish under 400 calories"* — and receive highly relevant, AI-curated recommendations with detailed nutritional and culinary information.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔍 **Semantic Search** | Find foods by meaning, not just keywords, using vector embeddings |
| 🌍 **Cuisine Filtering** | Filter results by cuisine type (Italian, Thai, Indian, Japanese, and more) |
| 🔥 **Calorie Filtering** | Enforce a strict calorie budget per meal using metadata filters |
| 🤖 **AI Chatbot (RAG)** | Conversational recommendations powered by IBM Granite LLM |
| 📊 **Comparison Mode** | Side-by-side AI analysis of two different food queries |
| 🧾 **Calorie Budget Checker** | See exactly which dishes fit (or exceed) your calorie limit |
| 🕑 **Search History** | Tracks your recent queries for easy revisiting |
| ⚡ **Performance Benchmarking** | Compare response times across all three search systems |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        User Interface Layer                      │
│   interactive_search.py  │  advanced_search.py  │  calorie_checker.py  │  enhanced_rag_chatbot.py  │
└─────────────────────────────────┬────────────────────────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │      shared_functions.py   │
                    │   (Core Utility Module)    │
                    └────────────┬──────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
 ┌────────▼────────┐   ┌─────────▼────────┐   ┌────────▼────────┐
 │  FoodDataSet.json│   │   ChromaDB       │   │ IBM watsonx.ai  │
 │  (Food Database)│   │ (Vector Store)   │   │ Granite 3.3 LLM │
 │                 │   │                  │   │                 │
 │ • food_name     │   │ • HNSW Index     │   │ • Context-aware │
 │ • ingredients   │   │ • Cosine Space   │   │   responses     │
 │ • cuisine_type  │   │ • Metadata       │   │ • Natural lang. │
 │ • calories      │   │   filtering      │   │   generation    │
 │ • health_info   │──►│                  │──►│                 │
 │ • taste_profile │   │ Embeddings via:  │   │                 │
 │ • cooking_method│   │ all-MiniLM-L6-v2 │   │                 │
 └─────────────────┘   └──────────────────┘   └─────────────────┘
```

---

## 📁 Project Structure

```
Food_Recommendation_System_Chroma_DB/
│
├── shared_functions.py       # Core module: data loading, ChromaDB setup, search utilities
├── interactive_search.py     # System 1: Simple CLI chatbot with search history
├── advanced_search.py        # System 2: Advanced filtering by cuisine & calories
├── calorie_checker.py        # System 3: Per-meal calorie budget enforcer
├── enhanced_rag_chatbot.py   # System 4: RAG chatbot powered by IBM Granite LLM
├── system_comparison.py      # Benchmarking tool: compare all systems side-by-side
│
└── FoodDataSet.json          # Food dataset (required at runtime — see Dataset section)
```

---

## 🛠️ Tech Stack

| Technology | Version | Purpose |
|---|---|---|
| **Python** | 3.8+ | Core programming language |
| **ChromaDB** | Latest | Vector database for semantic search |
| **Sentence Transformers** | Latest | `all-MiniLM-L6-v2` for text embeddings |
| **IBM watsonx.ai SDK** | Latest | IBM Granite 3.3 8B Instruct LLM |
| **NumPy** | Latest | Numerical operations |

---

## ✅ Prerequisites

Before running this project, ensure you have:

- **Python 3.8 or higher** installed
- **IBM watsonx.ai account** (required only for `enhanced_rag_chatbot.py`)
  - A valid `project_id` (use `"skills-network"` for IBM Skills Network environments)
  - Access to the `us-south.ml.cloud.ibm.com` endpoint
- A `FoodDataSet.json` file in the project root (see [Dataset](#-dataset))

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Bharath-kumar-7/Food_Recommendation_System_Chroma_DB.git
cd Food_Recommendation_System_Chroma_DB
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv

# On macOS/Linux
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install chromadb sentence-transformers ibm-watsonx-ai numpy
```

---

## 📦 Dataset

The system requires a `FoodDataSet.json` file placed in the project root directory. Each entry in the JSON array should follow this schema:

```json
[
  {
    "food_id": "1",
    "food_name": "Margherita Pizza",
    "food_description": "Classic Italian pizza with tomato sauce and mozzarella",
    "food_ingredients": ["flour", "tomato sauce", "mozzarella", "basil", "olive oil"],
    "cuisine_type": "Italian",
    "cooking_method": "Baked",
    "food_calories_per_serving": 285,
    "food_health_benefits": "Good source of calcium and lycopene",
    "food_nutritional_factors": {
      "protein": "12g",
      "carbs": "36g",
      "fat": "10g"
    },
    "food_features": {
      "taste": "savory",
      "texture": "crispy",
      "spice_level": "mild"
    }
  }
]
```

**Supported cuisine types:** Italian · Thai · Mexican · Indian · Japanese · French · Mediterranean · American · Health Food · Dessert

---

## 💻 Usage

### 1. Interactive Search

A simple CLI chatbot for natural language food search with search history.

```bash
python interactive_search.py
```

**Available commands:**

| Command | Description |
|---|---|
| Any text | Search for foods matching your query |
| `history` | View your last 10 searches |
| `help` | Show the help menu |
| `quit` / `exit` | Exit the application |

**Example:**
```
🔍 Search for food: spicy Thai noodles

✅ Found 5 recommendations:
════════════════════════════════════════════════════════════
1. 🍽️  Pad Thai
   📊 Match Score: 94.2%
   🏷️  Cuisine: Thai
   🔥 Calories: 380 per serving
   📝 Description: Classic Thai stir-fried noodles with tamarind sauce
```

---

### 2. Advanced Filtered Search

Demonstrates powerful metadata filtering by combining semantic search with cuisine type and calorie constraints.

```bash
python advanced_search.py
```

**Menu options:**

| Option | Description |
|---|---|
| `1` | Basic similarity search |
| `2` | Cuisine-filtered search |
| `3` | Calorie-filtered search (e.g., under 300 cal) |
| `4` | Combined filters (cuisine + calories) |
| `5` | Run pre-built demonstration scenarios |
| `6` | Help |
| `7` | Exit |

**Example — Low-calorie Italian search:**
```
Enter search query: pasta
Enter cuisine number: 1  (Italian)
Enter maximum calories: 400

🔍 Searching for 'pasta' in Italian cuisine with max 400 calories...
📋 Cuisine-Filtered Results (Italian)
══════════════════════════════════════════════════

1. 🍽️  Spaghetti Aglio e Olio
   📊 Similarity Score: 91.3%
   🏷️  Cuisine: Italian
   🔥 Calories: 320
   📝 Description: Light garlic and olive oil pasta
```

---

### 3. Calorie Checker

Helps users identify which foods fit within a defined per-meal calorie budget.

```bash
python calorie_checker.py
```

**Example session:**
```
💪 What's your calorie budget per meal? 350

🎯 Your calorie budget: 350 calories

🔍 Search for a food (or 'quit' to exit): grilled chicken

✅ FITS YOUR BUDGET (350 cal limit):
  1. Grilled Chicken Salad
     Calories: 280 (🟢 70 cal remaining)
     Cuisine: Health Food

🚫 OVER BUDGET (but similar to your search):
  1. Chicken Tikka Masala
     Calories: 420 (🔴 70 cal over budget)
```

---

### 4. Enhanced RAG Chatbot

> ⚠️ **Requires IBM watsonx.ai credentials.** Configure `my_credentials` and `project_id` in `enhanced_rag_chatbot.py` before running.

An AI-powered conversational chatbot that combines ChromaDB vector search (retrieval) with IBM Granite 3.3 LLM (generation) for contextually rich recommendations.

```bash
python enhanced_rag_chatbot.py
```

**Commands:**

| Command | Description |
|---|---|
| Any natural language text | Get AI-powered food recommendations |
| `compare` | AI comparison of two different food queries |
| `help` | Show detailed help and tips |
| `quit` | Exit the chatbot |

**Example conversation:**
```
👤 You: I want something spicy and healthy for dinner tonight

🔍 Searching vector database for: 'I want something spicy and healthy for dinner'...
✅ Found 3 relevant matches
🧠 Generating AI-powered response...

🤖 Bot: Great choice! For a spicy yet healthy dinner, I'd recommend
        the Chicken Tikka Masala — it's packed with protein, features
        anti-inflammatory spices like turmeric and ginger, and comes in
        at 380 calories. Alternatively, Thai Basil Stir-Fry offers a
        lighter option at 290 calories with a bold, fresh heat from
        Thai chilies. Both are excellent choices if you're looking for
        flavor without compromising nutrition!
```

---

### 5. System Comparison

Benchmarks all three search approaches against the same query and displays a side-by-side performance summary.

```bash
python system_comparison.py
```

**Sample output:**
```
📊 SYSTEM COMPARISON SUMMARY:
══════════════════════════════════════════════════
Interactive Search:
  ✅ Fast and simple
  ✅ Direct results display
  ❌ Limited context

Advanced Search:
  ✅ Powerful filtering options
  ✅ Multiple search modes
  ✅ Precise control
  ❌ Requires user to know filter options

RAG Chatbot:
  ✅ Natural language interaction
  ✅ Contextual explanations
  ✅ Conversational experience
  ❌ More complex implementation

⏱️ Performance Comparison:
  Interactive: 0.124s
  Advanced:    0.198s
  RAG Chatbot: 0.143s
```

---

## ⚙️ How It Works

### 1. Data Ingestion
Food items from `FoodDataSet.json` are loaded and normalised. A rich text document is constructed for each food item, combining its name, description, ingredients, cuisine type, cooking method, taste profile, health benefits, and nutritional factors.

### 2. Vector Embedding
Each text document is converted into a dense numerical vector using the **`all-MiniLM-L6-v2`** Sentence Transformer model. These vectors capture the semantic meaning of the food description.

### 3. ChromaDB Indexing
Embeddings are stored in a **ChromaDB** collection using a **cosine similarity HNSW index** alongside rich metadata (calories, cuisine, cooking method, etc.) that enables fast, filterable retrieval.

### 4. Semantic Search
When a user submits a query, it is embedded using the same model. ChromaDB finds the nearest vectors in the index (i.e., the most semantically similar foods), optionally filtered by cuisine type or calorie constraints.

### 5. RAG Response Generation (Enhanced Chatbot)
The top matching food items are used as **context** and injected into a structured prompt sent to **IBM Granite 3.3 8B Instruct** via IBM watsonx.ai. The LLM generates a natural, conversational response grounded in the retrieved food data, explaining why each recommendation matches the user's request.

```
User Query
    │
    ▼
Embed query (Sentence Transformer)
    │
    ▼
ChromaDB similarity search (+ optional metadata filters)
    │
    ▼
Top-K food items retrieved
    │
    ▼
Build structured prompt with retrieved context
    │
    ▼
IBM Granite 3.3 LLM generates natural language response
    │
    ▼
Display AI response + raw search result details
```

---

## 📚 Module Reference

### `shared_functions.py`

The core utility module used by all other scripts.

| Function | Description |
|---|---|
| `load_food_data(file_path)` | Load and normalise food items from JSON |
| `create_similarity_search_collection(name, metadata)` | Create a ChromaDB collection with sentence transformer embeddings |
| `populate_similarity_collection(collection, food_items)` | Embed and index all food items in ChromaDB |
| `perform_similarity_search(collection, query, n_results)` | Run a basic semantic similarity search |
| `perform_filtered_similarity_search(collection, query, cuisine_filter, max_calories, n_results)` | Semantic search with optional cuisine and calorie filters |

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. **Fork** this repository
2. **Create** a feature branch: `git checkout -b feature/your-feature-name`
3. **Commit** your changes: `git commit -m "Add your feature"`
4. **Push** to your fork: `git push origin feature/your-feature-name`
5. **Open** a Pull Request describing your changes

Please ensure your code follows the existing style and includes relevant comments.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ using ChromaDB, Sentence Transformers & IBM watsonx.ai

</div>
