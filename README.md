# loreai
A locally executable RPG Lore Management LLM

Step-by-step installation for Windows

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
