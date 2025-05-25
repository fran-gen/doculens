# doculens
A multimodal Streamlit app that lets you query PDFs using vision-enabled LLMs like Claude 3.5 and GPT-4o, powered by a ColPaLi (vidore/colpali-v1.2) retrieval engine.

---

## 🚀 Features

- 🖼 Converts PDFs to images  
- 🧠 Multimodal indexing with RAG 
- 💬 Generate answers based on your indexed document

---

## 📦 Requirements

- **Python** 3.12+  
- [**Poetry**](https://python-poetry.org/docs/#installation)  
- **Poppler** (for PDF processing)

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/fran-gen/doculens.git
cd doculens
```

### 2. Install dependencies with Poetry

```bash
poetry install
```

### 3. Activate the virtual environment

```bash
poetry shell
```

---

## 🧩 Install Poppler (Required for `pdf2image`)

### Ubuntu/Debian

```bash
sudo apt update
sudo apt install poppler-utils
```

### macOS (with Homebrew)

```bash
brew install poppler
```

### Windows

1. Download Poppler from: https://github.com/oschwartz10612/poppler-windows/releases/  
2. Extract it (e.g., to `C:\poppler`)  
3. Add `C:\poppler\bin` to your system PATH  
4. Or specify `poppler_path` in code:

---

## 🧪 Usage

To run the indexing pipeline:

```bash
python doculens/multimodal_app.py
```