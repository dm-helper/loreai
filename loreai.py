
import argparse
import os
import sys
import uuid
import datetime
from pathlib import Path
from typing import List, Dict, Any

# --- Embeddings & Vector Store ---
import chromadb
from chromadb.utils import embedding_functions

# --- Optional, used for nicer chunking if available ---
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    RecursiveCharacterTextSplitter = None

# --- Local LLM (Ollama) ---
try:
    import ollama
except ImportError:
    ollama = None


DEFAULT_MODEL = os.environ.get("DNDAI_MODEL", "llama3")
BASE_DIR = Path(os.environ.get("DNDAI_HOME", "./loreai")).resolve()
CANON_DIR = BASE_DIR / "lore" / "canon"
WORK_DIR = BASE_DIR / "lore" / "workbench"
DB_DIR = BASE_DIR / "vector_db"

SYSTEM_PROMPT = """You are a Dungeons & Dragons world-building assistant.
You answer questions using ONLY the provided CONTEXT when possible.
When you invent new lore, present it with concise, well-structured text suitable for adding to canon.
Avoid contradicting existing context and keep names consistent and sensible.

Always keep answers compact matching the mood and tone of the provided THEME.
"""

THEME_FILE = BASE_DIR / "theme.md"

def load_theme(theme_file: str = None) -> str:
    """Load default tone/theme from a given file, or fallback to theme.md."""
    if theme_file:
        path = Path(theme_file)
        if not path.is_absolute():
            path = BASE_DIR / path
    else:
        path = BASE_DIR / "theme.md"

    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return f"(No theme file found at {path})"


def ensure_dirs():
    CANON_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)


def load_text_files(folder: Path) -> List[Dict[str, Any]]:
    docs = []
    for p in folder.rglob("*"):
        if p.suffix.lower() in {".md", ".txt"} and p.is_file():
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
            docs.append({"path": str(p), "text": txt})
    return docs


def chunk_text(text: str, source: str, chunk_size: int = 1200, chunk_overlap: int = 150) -> List[Dict[str, Any]]:
    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n# ", "\n\n", "\n", " "],
        )
        chunks = splitter.split_text(text)
    else:
        # Simple fallback chunker by characters
        chunks = []
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunks.append(text[start:end])
            start = end - chunk_overlap
            if start < 0:
                start = 0
    out = []
    for i, ch in enumerate(chunks):
        out.append(
            {
                "id": f"{uuid.uuid4()}",
                "text": ch,
                "metadata": {"source": source, "chunk": i},
            }
        )
    return out


def get_collection():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    coll = client.get_or_create_collection(name="dnd_lore", embedding_function=embedder, metadata={"hnsw:space": "cosine"})
    return coll


def ingest_folder(folder: Path):
    ensure_dirs()
    folder = folder.resolve()
    if not folder.exists():
        print(f"[!] Folder not found: {folder}")
        return
    docs = load_text_files(folder)
    if not docs:
        print(f"[i] No .md or .txt files found in {folder}")
        return

    coll = get_collection()
    to_add_ids, to_add_texts, to_add_meta = [], [], []
    for d in docs:
        chunks = chunk_text(d["text"], d["path"])
        for ch in chunks:
            to_add_ids.append(ch["id"])
            to_add_texts.append(ch["text"])
            to_add_meta.append(ch["metadata"])

    coll.add(ids=to_add_ids, documents=to_add_texts, metadatas=to_add_meta)
    print(f"[*] Ingested {len(to_add_texts)} chunks from {len(docs)} files.")


def ingest_file(file_path: Path, global_core: bool = False, regional_core: str = None):
    """Ingest a single file into vector DB with optional core metadata.
    Always clears existing entries for this file first to avoid duplication.
    """
    ensure_dirs()
    if not file_path.exists():
        print(f"[!] File not found: {file_path}")
        return

    coll = get_collection()

    # === Safe ingestion: delete old entries first ===
    coll.delete(where={"source": str(file_path)})

    # Read and chunk file
    text = file_path.read_text(encoding="utf-8")
    chunks = chunk_text(text, str(file_path))

    ids, texts, metas = [], [], []
    for ch in chunks:
        ids.append(ch["id"])
        texts.append(ch["text"])
        meta = ch["metadata"]
        meta["global_core"] = global_core
        if regional_core:
            meta["regional_core"] = regional_core.lower()
        metas.append(meta)

    # Add fresh chunks
    coll.add(ids=ids, documents=texts, metadatas=metas)

    print(f"[*] Re-ingested {len(chunks)} chunks from {file_path.name} "
          f"(global_core={global_core}, regional_core={regional_core})")




def query_lore(question: str, k: int = 8, region: str = None) -> Dict[str, Any]:
    """
    Query the lore database for relevant context.
    Includes:
      - similarity search results for the given question
      - all global-core lore
      - all regional-core lore (if region is specified)
    Returns a dict with combined context text, docs, and metadata.
    """
    ensure_dirs()
    coll = get_collection()

    # Always retrieve global core (using dummy query text to satisfy Chroma)
    core_res = coll.query(
        query_texts=["core context"],
        where={"global_core": True},
        n_results=1000
    )
    core_docs = core_res.get("documents", [[]])[0]
    core_metas = core_res.get("metadatas", [[]])[0]

    # Optionally retrieve regional core
    regional_docs, regional_metas = [], []
    if region:
        reg_res = coll.query(
            query_texts=["regional context"],
            where={"regional_core": region.lower()},
            n_results=1000
        )
        regional_docs = reg_res.get("documents", [[]])[0]
        regional_metas = reg_res.get("metadatas", [[]])[0]

    # Retrieve query-specific chunks
    res = coll.query(query_texts=[question], n_results=k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]

    # Merge all results
    combined_docs = core_docs + regional_docs + docs
    combined_metas = core_metas + regional_metas + metas

    # Assemble context string for the LLM
    context = ""
    for i, d in enumerate(combined_docs):
        src = combined_metas[i].get("source", "unknown")
        context += f"\n[Source: {src}]\n{d}\n"

    return {"context": context, "docs": combined_docs, "metas": combined_metas}



def call_llm(messages: List[Dict[str, str]], model: str = DEFAULT_MODEL) -> str:
    if ollama is None:
        raise RuntimeError("The 'ollama' Python package is not installed. Install with: pip install ollama")
    try:
        response = ollama.chat(model=model, messages=messages)
        return response["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Ollama chat failed: {e}")



def ask(question: str, model: str = DEFAULT_MODEL, k: int = 8, theme_name: str = None, region: str = None) -> str:
    retrieved = query_lore(question, k=k, region=region)
    context = retrieved["context"]
    theme_text = load_theme(theme_name)
    user_prompt = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"Provide elaborate answers. When asked about groups, locations, or organizations, always provide a **complete list of all relevant known entities** from the context. If multiple individuals are associated, list each with all available details.\n\n"
        f"Follow SYSTEM_PROMPT rules.\n"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    print("\n=== AI QUERY ===\n")
    print(messages)
    print("\n=== END OF INPUT ===\n")
    answer = call_llm(messages, model=model)
    print("\n=== AI GENERATED LORE ===\n")
    print(answer)
    print("\n=== END OF OUTPUT ===\n")
    return answer



def create_new(prompt: str, model: str = DEFAULT_MODEL, theme_name: str = None):
    """
    Generate new lore, cache it, and offer interactive options:
    Draft, Publish, Discard, or Revise.
    Each revision includes the previous AI output as context.
    """
    guidance = (
        "Generate new lore aligned with the existing lore and THEME.\n"
        "When creating NPCs, always include the following fields: "
        "Name, Race, Class / Profession, Role / Occupation, Motivations / Goals, Affiliations / Organizations, Key Relationships, Description.\n"
        "- If multiple NPCs are requested, output each NPC in a clearly separated block, including all fields and provide a thorough description.\n"
        "- Keep answers concise and consistent with the existing lore.\n"
        "- Use the following template for each NPC:\n\n"
        "NPC Template:\n"
        "Name:\n"
        "Race:\n"
        "Class/Profession/Role/Occupation:\n"
        "Description:\n"
        "Motivations & Goals:\n"
        "Affiliations:\n"
        "Key Relationships:\n"
    )
    theme_text = load_theme(theme_name)

    current_prompt = prompt
    previous_answer = ""

    while True:
        # Build prompt including previous answer if present
        full_prompt = f"{guidance}\n\nTHEME:\n{theme_text}\n\nUSER PROMPT: {current_prompt}"
        if previous_answer:
            full_prompt += f"\n\nPREVIOUS AI OUTPUT:\n{previous_answer}\n\nPlease revise or expand based on the above."

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": full_prompt},
        ]

        print("\n=== AI QUERY ===\n")
        print(messages)
        print("\n=== END OF INPUT ===\n")
        answer = call_llm(messages, model=model)
        cache_last(answer)
        print("\n=== AI GENERATED LORE ===\n")
        print(answer)
        print("\n=== END OF OUTPUT ===\n")

        # Save previous answer for next iteration
        previous_answer = answer

        # Interactive prompt
        choice = input("[R]evise / [D]raft / [P]ublish / [X] discard? ").strip().upper()
        if choice == "D":
            title = input("Enter draft title: ").strip()
            draft_to_workdir(title=title)
            print("[*] Draft saved.")
            break
        elif choice == "P":
            title = input("Enter canon title: ").strip()
            try:
                publish_to_canon(title=title)
                print("[*] Published to canon.")
            except FileExistsError as e:
                print(f"[!!!] {e}")
            break
        elif choice == "R":
            # Revise prompt; next iteration includes previous answer
            current_prompt = input("Enter revised prompt: ").strip()
            print("[i] Generating revised output...\n")
            continue
        elif choice == "X":
            print("[i] Discarded output.")
            break
        else:
            print("[!!!] Invalid choice. Enter R, D, P or X")

    return answer



def cache_last(text: str):
    cache_dir = BASE_DIR / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / "last_answer.md", "w", encoding="utf-8") as f:
        f.write(text)



def draft_to_workdir(title: str, content_path: Path = None, generated_text: str = None):
    """
    Save AI-generated lore (or existing text) into WORK_DIR as a draft.
    Defaults to last cached answer if nothing provided.
    Drafts are not ingested into the vector DB until published.
    """
    cache_path = BASE_DIR / ".cache" / "last_answer.md"

    if content_path is not None:
        text = Path(content_path).read_text(encoding="utf-8")
    elif generated_text is not None:
        text = generated_text
    elif cache_path.exists():
        text = cache_path.read_text(encoding="utf-8")
    else:
        raise ValueError("Either content_path, generated_text, or cached last_answer.md must be available.")

    # Create safe filename
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    safe_title = title.lower().replace(" ", "-")
    draft_filename = f"{timestamp}-{safe_title}.md"
    draft_file_path = WORK_DIR / draft_filename

    draft_file_path.write_text(text, encoding="utf-8")
    print(f"[*] Draft saved: {draft_file_path}")



def publish_to_canon(title: str, content_path: Path = None, generated_text: str = None, global_core: bool = False, regional_core: str = None):
    """
    Publish content into canon, update references in related files,
    and ingest it into the vector DB.
    Uses cached last_answer.md if no content provided.
    Refuses to overwrite if a file with the same title already exists.
    """
    cache_path = BASE_DIR / ".cache" / "last_answer.md"

    if content_path is not None:
        text = Path(content_path).read_text(encoding="utf-8")
    elif generated_text is not None:
        text = generated_text
    elif cache_path.exists():
        text = cache_path.read_text(encoding="utf-8")
    else:
        raise ValueError("No content provided. Use --content_path, --generated_text, or run 'create' first.")

    # Create clean filename (no timestamp)
    safe_title = title
    canon_filename = f"{safe_title}.md"
    canon_file_path = CANON_DIR / canon_filename

    if canon_file_path.exists():
        # Append cleanly, just a blank line before the new content
        with open(canon_file_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n{content}\n")
        print(f"[*] Appended new lore to existing file: {canon_file_path}")
    else:
        # Create a new file
        canon_file_path.write_text(content + "\n", encoding="utf-8")
        print(f"[*] Published new lore: {canon_file_path}")

    # Detect relevant existing files
    relevant_files = []
    for md_file in CANON_DIR.glob("*.md"):
        if md_file == canon_file_path:
            continue
        if md_file.stem.lower() in text.lower():
            relevant_files.append(md_file)

    # Update References
    for existing_file in relevant_files:
        append_reference_to_file(existing_file, canon_file_path)
        print(f"[*] Added reference to {canon_file_path.name} in {existing_file.name}")
 
    # === Safe ingestion: delete old entries first, then re-add ===
    coll = get_collection()
    coll.delete(where={"source": str(canon_file_path)})

    file_text = canon_file_path.read_text(encoding="utf-8")
    chunks = chunk_text(file_text, str(canon_file_path))
    for ch in chunks:
        if global_core:
            ch["metadata"]["global_core"] = True
        if regional_core:
            ch["metadata"]["regional_core"] = regional_core.lower()

    coll.add(
        ids=[ch["id"] for ch in chunks],
        documents=[ch["text"] for ch in chunks],
        metadatas=[ch["metadata"] for ch in chunks],
    )

    print(f"[*] Re-ingested {len(chunks)} chunks from {canon_file_path.name} into vector DB")


def mark_core(file: str, global_core: bool = False, regional_core: str = None):
    """Mark an existing canon file as global or regional core by re-ingesting with metadata."""
    path = Path(file)
    if not path.is_absolute():
        path = CANON_DIR / path
        if not path.suffix:
            path = path.with_suffix(".md")

    if not path.exists():
        print(f"[!] File not found: {path}")
        return

    ingest_file(path, global_core=global_core, regional_core=regional_core)
    print(f"[*] Marked {path.name} as "
          f"{'global-core' if global_core else ''} "
          f"{'regional-core='+regional_core if regional_core else ''}".strip())


def list_drafts(return_list=False):
    """List all drafts in workbench, newest first. 
    If return_list=True, return Path objects instead of printing."""
    drafts = sorted(WORK_DIR.glob("*.md"), key=os.path.getmtime, reverse=True)
    if return_list:
        return drafts

    from datetime import datetime

    print("\n=== Drafts ===")
    if not drafts:
        print("(none)")
    else:
        for i, f in enumerate(drafts, 1):
            ts = datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{i}] {f.name}  (saved {ts})")
    print("================\n")


def delete_draft(index: int):
    """Delete a single draft by index, with confirmation."""
    drafts = list_drafts(return_list=True)
    if index < 1 or index > len(drafts):
        print(f"Invalid draft index {index}")
        return
    target = drafts[index - 1]

    confirm = input(f"Are you sure you want to delete draft '{target.name}'? (y/N): ").strip().lower()
    if confirm == "y":
        target.unlink()
        print(f"[*] Deleted draft: {target.name}")
    else:
        print("[*] Cancelled.")


def clear_drafts():
    """Delete all drafts from the workbench, with confirmation."""
    drafts = list_drafts(return_list=True)
    if not drafts:
        print("[*] No drafts to delete.")
        return

    confirm = input(f"Are you sure you want to delete ALL {len(drafts)} drafts? (y/N): ").strip().lower()
    if confirm == "y":
        for d in drafts:
            d.unlink()
        print(f"[*] Cleared {len(drafts)} drafts from {WORK_DIR}")
    else:
        print("[*] Cancelled.")



def unmark_core(file: str):
    """Remove core metadata from an existing file by re-ingesting without core flags."""
    path = Path(file)
    if not path.is_absolute():
        path = CANON_DIR / path
        if not path.suffix:
            path = path.with_suffix(".md")

    if not path.exists():
        print(f"[!] File not found: {path}")
        return

    ingest_file(path, global_core=False, regional_core=None)
    print(f"[*] Removed core status from {path.name}")


def list_core():
    """List all files marked as global or regional core, grouped by type."""
    coll = get_collection()
    results = coll.get(include=["metadatas"], where={})

    global_core_files = []
    regional_core_files = {}

    for meta in results["metadatas"]:
        if not meta:
            continue
        source = meta.get("source")
        if meta.get("global_core", False):
            if source not in global_core_files:
                global_core_files.append(source)
        regional = meta.get("regional_core")
        if regional:
            if regional not in regional_core_files:
                regional_core_files[regional] = []
            if source not in regional_core_files[regional]:
                regional_core_files[regional].append(source)

    print("\n=== CORE FILES ===\n")
    if global_core_files:
        print("Global Core:")
        for f in sorted(set(global_core_files)):
            print(f"  - {Path(f).name}")
    else:
        print("Global Core: (none)")

    if regional_core_files:
        print("\nRegional Core:")
        for region, files in regional_core_files.items():
            print(f"  {region}:")
            for f in sorted(set(files)):
                print(f"    - {Path(f).name}")
    else:
        print("\nRegional Core: (none)")

    print("\n==================\n")



def remove_lore(file_path: str):
    path = Path(file_path)
    if not path.is_absolute():
        path = CANON_DIR / path
    if not path.exists():
        print(f"[!] File not found: {path}")
        return

    # Delete the file
    path.unlink()
    print(f"[?] Deleted file: {path}")

    # Remove from vector DB
    coll = get_collection()
    coll.delete(where={"source": str(path)})
    print(f"[?] Removed entries from vector DB")


def reset_model():
    # Clear vector DB
    if DB_DIR.exists():
        import shutil
        shutil.rmtree(DB_DIR)
        print("[?] Vector DB cleared")

    # Clear cache
    cache_dir = BASE_DIR / ".cache"
    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
        print("[?] Cache cleared")


def export_lore(out_file: str):
    ensure_dirs()
    all_text = ""
    for md_file in sorted(CANON_DIR.glob("*.md")):
        all_text += f"\n\n# {md_file.stem}\n\n"
        all_text += md_file.read_text(encoding="utf-8")
    out_path = Path(out_file)
    out_path.write_text(all_text, encoding="utf-8")
    print(f"[?] Exported all lore to {out_path}")


def backup_model(out_dir: str):
    import shutil
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    shutil.copytree(CANON_DIR, out_path / "canon", dirs_exist_ok=True)
    shutil.copytree(DB_DIR, out_path / "vector_db", dirs_exist_ok=True)
    print(f"[?] Backup created at {out_path}")


def append_reference_to_file(existing_file: Path, reference_file: Path):
    """
    Add reference_file to the References section of existing_file.
    If References section doesn't exist, create it at the bottom.
    """
    content = existing_file.read_text(encoding="utf-8")
    ref_line = f"- {reference_file.name}"

    if "## References" in content:
        base, refs = content.split("## References", 1)
        current_refs = [r.strip("- ") for r in refs.strip().splitlines()]
        if ref_line not in [f"- {r}" for r in current_refs]:
            updated_refs = "\n".join(current_refs + [ref_line])
            existing_file.write_text(f"{base}## References\n{updated_refs}", encoding="utf-8")
    else:
        # Create new References section
        existing_file.write_text(content + f"\n\n## References\n{ref_line}", encoding="utf-8")



def detect_relevant_files(new_content: str) -> list[Path]:
    relevant = []
    for md_file in CANON_DIR.glob("*.md"):
        if md_file.stem.lower() in new_content.lower():
            relevant.append(md_file)
    return relevant



def bootstrap_example():
    """Creates starter folders and a sample lore file."""
    ensure_dirs()
    sample = CANON_DIR / "world.md"
    if not sample.exists():
        sample.write_text(
            "# World Overview\n\n"
            "The world of Thalrune features the coastal city of Starmantle, dwarven clans like the Strongarms, "
            "and shifting alliances between guilds and temples.\n",
            encoding="utf-8",
        )
        ingest_folder(CANON_DIR)
        print(f"[*] Created sample lore at {sample}")
    else:
        print("[i] Sample lore already exists.")


def main():
    parser = argparse.ArgumentParser(
        description="LoreAI Lore Assistant (Local, Windows-friendly D&D Worldbuilding Assistant)"
    )
    sub = parser.add_subparsers(dest="cmd")

    # --- Ingest ---
    p_ingest = sub.add_parser("ingest", help="Ingest a folder or single .md/.txt file into the vector DB")
    p_ingest.add_argument("path", type=str, help="Path to folder or file")
    p_ingest.add_argument("--global-core", action="store_true", help="Mark this lore as global core")
    p_ingest.add_argument("--regional-core", type=str, default=None, help="Mark this lore as regional core (region name)")

    # --- Ask ---
    p_ask = sub.add_parser("ask", help="Ask a question against your lore")
    p_ask.add_argument("question", type=str, nargs="+", help="Your question text")
    p_ask.add_argument("--k", type=int, default=8, help="Number of retrieved chunks")
    p_ask.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama model name (e.g., llama3)")
    p_ask.add_argument("--theme", type=str, default=None, help="Path to theme file")
    p_ask.add_argument("--region", type=str, default=None, help="Optional region tag to include regional core lore")

    # --- Create ---
    p_create = sub.add_parser("create", help="Create new lore (interactive). After generation, you can Draft, Publish, Revise, or Discard.")
    p_create.add_argument("prompt", type=str, nargs="+", help="Prompt for new lore")
    p_create.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p_create.add_argument("--theme", type=str, default=None, help="Path to theme file (e.g., themes/epic-style.md)")

    # --- Draft ---
    p_draft = sub.add_parser("draft", help="Save last AI output (or a given file) as a draft in the workbench. Timestamped filename created.")
    p_draft.add_argument("--title", type=str, required=True, help="Title for the draft file")
    p_draft.add_argument("--file", type=str, default=None, help="Optional file path to save instead of last answer")

    # --- Publish ---
    p_publish = sub.add_parser("publish", help="Publish last AI output (or a given file) to canon. Refuses to overwrite existing titles.")
    p_publish.add_argument("--title", type=str, required=True, help="Title for the canon file")
    p_publish.add_argument("--file", type=str, default=None, help="Optional file path to publish instead of last answer")
    p_publish.add_argument("--draft", type=int, help="Index of draft from list-drafts")
    p_publish.add_argument("--global-core", action="store_true", help="Mark this lore as global core (always included)")
    p_publish.add_argument("--regional-core", type=str, default=None, help="Mark this lore as regional core (tagged by region name)")

    # --- Draft Management ---
    sub.add_parser("list-drafts", help="List all draft files in the workbench")

    p_del_draft = sub.add_parser("delete-draft", help="Delete a specific draft by index")
    p_del_draft.add_argument("index", type=int, help="Draft index from list-drafts")

    sub.add_parser("clear-drafts", help="Delete ALL drafts from the workbench")

    # --- Mark Core ---
    p_mark = sub.add_parser("mark-core", help="Mark a canon file as global or regional core")
    p_mark.add_argument("file", type=str, help="Path (or title) of the canon file")
    p_mark.add_argument("--global-core", action="store_true", help="Mark as global core")
    p_mark.add_argument("--regional-core", type=str, default=None, help="Mark as regional core (region name)")

    # --- Unmark Core ---
    p_unmark = sub.add_parser("unmark-core", help="Remove core status from a canon file")
    p_unmark.add_argument("file", type=str, help="Path (or title) of the canon file")

    # --- List Core ---
    sub.add_parser("list-core", help="List all files marked as global or regional core")

    # --- Remove ---
    p_remove = sub.add_parser("remove", help="Remove a lore file and its vector entries")
    p_remove.add_argument("file", type=str, help="Path to the lore file to remove")

    # --- Reset ---
    sub.add_parser("reset", help="Reset the assistant (clear vector DB and cache)")

    # --- Export ---
    p_export = sub.add_parser("export", help="Export all canon lore into a single markdown file")
    p_export.add_argument("--out", type=str, default="all_lore_export.md", help="Output file path")

    # --- Backup ---
    p_backup = sub.add_parser("backup", help="Backup lore and vector DB")
    p_backup.add_argument("--out", type=str, default=str(BASE_DIR / "backup"), help="Backup folder path")

    # --- Bootstrap ---
    sub.add_parser("bootstrap", help="Create starter folders and a sample lore file")

    args = parser.parse_args()

    if args.cmd == "ingest":
        path = Path(args.path)
        if path.is_file():
            ingest_file(path, global_core=args.global_core, regional_core=args.regional_core)
        elif path.is_dir():
            ingest_folder(path)
        else:
            print(f"[!] Path not found: {path}")
    elif args.cmd == "ask":
        question = " ".join(args.question)
        ask(question, model=args.model, k=args.k, theme_name=args.theme, region=args.region)
    elif args.cmd == "create":
        prompt = " ".join(args.prompt)
        create_new(prompt, model=args.model, theme_name=args.theme)
    elif args.cmd == "draft":
        draft_to_workdir(title=args.title, content_path=Path(args.file) if args.file else None)
    elif args.cmd == "publish":
        if args.draft:
            drafts = list_drafts(return_list=True)
            if args.draft < 1 or args.draft > len(drafts):
                print(f"Invalid draft index {args.draft}")
                return
            content_path = drafts[args.draft - 1]
        else:
            content_path = Path(args.file) if args.file else None
        publish_to_canon(title=args.title, content_path=content_path, global_core=args.global_core, regional_core=args.regional_core)
    elif args.cmd == "list-drafts":
        list_drafts()
    elif args.cmd == "delete-draft":
        delete_draft(args.index)
    elif args.cmd == "clear-drafts":
        clear_drafts()
    elif args.cmd == "mark-core":
        mark_core(args.file, global_core=args.global_core, regional_core=args.regional_core)
    elif args.cmd == "unmark-core":
        unmark_core(args.file)
    elif args.cmd == "list-core":
        list_core()
    elif args.cmd == "remove":
        remove_lore(args.file)
    elif args.cmd == "reset":
        reset_model()
    elif args.cmd == "export":
        export_lore(args.out)
    elif args.cmd == "backup":
        backup_model(args.out)
    elif args.cmd == "bootstrap":
        bootstrap_example()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
