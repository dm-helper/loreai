# loreai
A locally executable RPG Lore Management LLM

### **Current CLI Commands & Features**

````markdown
# 📖 LoreAI CLI Documentation

Run the assistant with:

```bash
python loreai.py <command> [options]
````

---

## 🔍 Ingest

**Description:** Ingests lore into the vector database so it can be used in queries.

**Usage:**
```bash
python loreai.py ingest <path> [--global-core] [--regional-core REGION]
```
The <path> can be either:
A folder containing multiple .md or .txt files
A single .md or .txt file

Options:
--global-core
Marks the ingested lore as global core, meaning it will always be included in responses.
--regional-core REGION
Marks the ingested lore as regional core for the specified region. Regional core lore is included in queries only when that region is active.

---

## ❓ Ask

**Description:** Query your canon lore. Only published canon (and core) files are used.

```bash
python loreai.py ask <question> [--k N] [--model MODEL] [--theme THEME] [--region REGION]
```

* `<question>` → your question text.
* `--k` (default 8) → number of chunks to retrieve.
* `--model` (default llama3) → Ollama model to use.
* `--theme` → optional theme file (e.g. `themes/epic-style.md`).
* `--region` → optional regional core tag to include region-specific lore.

---

## ✨ Create

**Description:** Generate new lore interactively. After generation, you can **Revise, Draft, Publish, or Discard**.

```bash
python loreai.py create <prompt> [--model MODEL] [--theme THEME]
```

* `<prompt>` → initial user prompt.
* `--model` → Ollama model.
* `--theme` → optional theme file.

---

## 📝 Drafts

### Save a Draft

Save the **last AI output** (or a given file) as a draft in the workbench. Drafts are timestamped.

```bash
python loreai.py draft --title <title> [--file PATH]
```

* `--title` → draft title.
* `--file` → optional file to save instead of last AI output.

### List Drafts

Show all current drafts with index numbers.

```bash
python loreai.py list-drafts
```

### Publish a Draft

Publish a draft by index from the draft list.

```bash
python loreai.py publish --draft <index> --title <canon-title> [--global-core] [--regional-core REGION]
```

* `<index>` → number from `list-drafts`.
* `--title` → title for the canon file (no overwrite allowed).
* `--global-core` → mark this lore as global core (always included).
* `--regional-core` → mark this lore as regional core (tagged by region name).

### Delete a Draft

Delete a single draft by index (with confirmation).

```bash
python loreai.py delete-draft <index>
```

### Clear All Drafts

Delete all drafts (with confirmation).

```bash
python loreai.py clear-drafts
```

---

## 📚 Publish

Publish the **last AI output** (or a given file) into canon. Updates references and ingests into vector DB.

```bash
python loreai.py publish --title <canon-title> [--file PATH] [--global-core] [--regional-core REGION]
```

* `--title` → canon file title (no overwrite).
* `--file` → optional file to publish instead of last AI output.
* `--global-core` → mark as global core.
* `--regional-core` → mark as regional core with region tag.

---

## 🔖 Core Management

### Mark Core

Mark a canon file as global or regional core.

```bash
python loreai.py mark-core <file> [--global-core] [--regional-core REGION]
```

### Unmark Core

Remove core status from a canon file.

```bash
python loreai.py unmark-core <file>
```

### List Core

List all files marked as global or regional core.

```bash
python loreai.py list-core
```

---

## 🗑️ Remove Lore

Remove a lore file and its vector DB entries.

```bash
python loreai.py remove <file>
```

---

## ⚙️ Reset

Reset the assistant by clearing the vector DB and cache.

```bash
python loreai.py reset
```

---

## 📤 Export

Export all canon lore into a single markdown file.

```bash
python loreai.py export [--out FILE]
```

* `--out` → output file path (default `all_lore_export.md`).

---

## 💾 Backup

Backup canon lore and vector DB.

```bash
python loreai.py backup [--out DIR]
```

* `--out` → backup folder path (default `<BASE_DIR>/backup`).

---

## 🚀 Bootstrap

Create starter folders and a sample lore file.

```bash
python loreai.py bootstrap
```

```
```


---

### **Interactive `create` Workflow Notes**
* Each iteration uses **previous AI output** if `Revise` is chosen.
* Drafts and canon publications use **separate functions**, so workflow is modular.
* Discarded output does **not affect cache** beyond that session.
* References section in canon files is automatically appended when publishing new content.

---

### **Step-by-step installation for Windows**

1) Install prerequisites
Python 3.11+ (Windows 10/11):
Download from python.org, check “Add Python to PATH” during install.
Ollama (local LLM runtime for Windows):
Download & install from ollama.ai (Windows build).
Open PowerShell and pull a model (good starter options):

ollama pull llama3

Test it:
ollama run llama3
(Type a message; Ctrl+C to stop.)

Tip: You can swap models later (e.g., mistral, llama3.1, etc.).

2) Get the files from github
Put them into a new folder, e.g. C:\Users\<you>\DndLoreAssistant\

3) Create a virtual environment & install deps
Open PowerShell in that folder and run:

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

If PowerShell blocks the activate script, run PowerShell as Administrator and:
Set-ExecutionPolicy RemoteSigned
then try activation again.

4) Bootstrap your project structure

Still in that folder (with venv active):
python loreai.py bootstrap

This creates:

loreai/
  lore/
    canon/       <-- permanent lore lives here
    workbench/   <-- drafts live here (optional)
  vector_db/     <-- persistent memory index

It also creates a sample world.md and indexes it.

5) Ingest your actual lore
Put your existing .md / .txt files in the canon folder
Then ingest them:
python loreai.py ingest "./loreai/lore/canon"

Run this any time you add/update files in your canon or other folders you want indexed.

6) Ask questions against your lore
python loreai.py ask "Who is Varrin Strongarm and what is his claim?"

The assistant retrieves the most relevant lore chunks and answers grounded in your canon.

7) Create new lore (not canon yet)
python loreai.py create "Invent a pirate faction in the southern seas with a unique calling card."

Nothing is permanent yet—this is exactly where you review and decide.

8) Save selected output into canon (permanent memory)
If you liked the last answer you saw (from ask or create), publish it:
python loreai.py publish --title "southern-sea-pirates"

This writes a file like:
loreai/lore/canon/-southern-sea-pirates.md
...and re-ingests canon so it’s immediately part of the knowledge base.

or store it for later
python loreai.py draft --title "southern-sea-pirates"

This writes a file like:
loreai/lore/workbench/2025-09-05-southern-sea-pirates.md

9) Where things live
Permanent memory: loreai/lore/canon/ + the persistent vector DB in loreai/vector_db/.

Working drafts: you can keep temporary text in loreai/lore/workbench/ (optional) and promote via publish --file ....

11) Swapping models or improving quality
Try different Ollama models:
ollama pull mistral
python loreai.py ask "..." --model mistral

Better retrieval: keep lore in smaller, focused markdown files with headers; the script automatically chunks large files.
