import re
import json
import os
import sys

from typing import List, Dict, Any, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.utils.task_utils import add_running_task
from app.import_process.agent.state import ImportGraphState
from app.core.logger import logger

# --- Configuration ---
DEFAULT_MAX_CONTENT_LENGTH = 2000
MIN_CONTENT_LENGTH = 500


def step_1_get_inputs(state: ImportGraphState) -> Tuple[Any, str, int]:
    """
    Step 1: Load and pre-process input data.
    Extracts MD content, file title, and max chunk length from the state dict, with basic normalisation.
    :param state: Pipeline state dict (ImportGraphState), must contain md_content and related keys
    :return: Tuple of (normalised MD content, file title, max chunk length);
             returns (None, None, None) if no valid MD content is present
    """
    # Extract raw MD content from state
    content = state.get("md_content")
    # Fallback: no MD content — terminate early
    if not content:
        logger.warning("No valid MD content in state, aborting document splitting")
        return None, None, None

    # Normalise line endings to avoid Windows/Linux differences causing downstream issues
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    # Use file title from state if available; default to "Unknown File"
    file_title = state.get("file_title", "Unknown File")
    # Use global default for max chunk length
    max_len = DEFAULT_MAX_CONTENT_LENGTH

    logger.info(f"Step 1: Input loaded — file title: {file_title}, max chunk length: {max_len}")
    return content, file_title, max_len


def step_2_split_by_titles(
    content: str, file_title: str
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Step 2: Initial split by Markdown headings.
    Splits the full MD document into sections by heading level (#–######),
    skipping headings inside code blocks to preserve semantic integrity.
    :param content: Normalised full MD content (string)
    :param file_title: Owning file title, used to tag each section
    :return: Tuple of (section list, valid heading count, total raw line count)
    """
    # Regex to match Markdown headings levels 1–6
    title_pattern = r'^\s*#{1,6}\s+.+'

    lines = content.split("\n")
    sections = []       # Final list of split sections
    current_title = ""  # Current section heading
    current_lines = []  # Line buffer for the current section
    title_count = 0     # Number of valid headings found
    in_code_block = False  # Flag to avoid treating # inside code blocks as headings

    def _flush_section():
        """Inner helper: write buffered lines as a section; skip if buffer is empty."""
        if not current_lines:
            return
        sections.append(
            {
                "title": current_title,
                "content": "\n".join(current_lines),
                "file_title": file_title,
            }
        )

    # Iterate line by line, detect headings and split into sections
    for line in lines:
        stripped_line = line.strip()
        # Detect code block boundaries (``` or ~~~): toggle flag on entry/exit
        if stripped_line.startswith("```") or stripped_line.startswith("~~~"):
            in_code_block = not in_code_block
            current_lines.append(line)
            continue

        # Valid heading: not inside a code block and matches heading pattern
        is_valid_title = (not in_code_block) and re.match(title_pattern, line)
        if is_valid_title:
            # New heading encountered: flush previous section, then start a new one
            _flush_section()
            current_title = line.strip()
            current_lines = [current_title]
            title_count += 1
            logger.debug(f"MD heading detected: {current_title}")
        else:
            # Regular line: append to current section buffer
            current_lines.append(line)

    # Flush the last buffered section after the loop ends
    _flush_section()
    logger.info(
        f"Step 2: Heading split complete — {title_count} heading(s) found, {len(lines)} raw lines"
    )
    return sections, title_count, len(lines)


def step_3_handle_no_title(
    content: str, sections: List[Dict[str, Any]], title_count: int, file_title: str
) -> List[Dict[str, Any]]:
    """
    Step 3: Fallback handling for documents with no headings.
    If no headings were detected, wraps the entire content as a single untitled section
    to prevent downstream logic from failing.
    :param content: Normalised full MD content
    :param sections: Section list produced by step 2
    :param title_count: Number of valid headings detected in step 2
    :param file_title: Owning file title
    :return: Section list (original or fallback single-section)
    """
    if title_count == 0:
        logger.warning(
            f"Step 3: No MD headings found, treating full document as a single section — file: {file_title}"
        )
        return [{"title": "Untitled", "content": content, "file_title": file_title}]
    logger.debug(f"Step 3: {title_count} heading(s) detected, no fallback needed")
    return sections


def _split_long_section(
    section: Dict[str, Any], max_length: int = DEFAULT_MAX_CONTENT_LENGTH
) -> List[Dict[str, Any]]:
    """
    Helper: secondary split for oversized sections using LangChain's recursive splitter.
    Splits from coarse to fine: paragraph → sentence → punctuation → space.
    Split priority: blank line (paragraph) → newline → CJK/EN punctuation → space → hard split.
    :param section: Source section dict; must contain 'content', optionally 'title'/'file_title'
    :param max_length: Max character length per chunk (defaults to global config)
    :return: List of sub-sections, each with parent title, index, and other metadata
    """
    # Fallback for empty content: return original section unchanged
    content = section.get("content", "") or ""
    # Within limit: no split needed — return as-is (list for consistent return type)
    if len(content) <= max_length:
        return [section]

    # Normalise line endings
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    title = section.get("title", "") or ""
    # Title prefix with blank line separator to distinguish from body text
    prefix = f"{title}\n\n" if title else ""
    # Available body length = total max - prefix length (title must not consume the entire chunk budget)
    available_len = max_length - len(prefix)
    # Edge case: title alone exceeds the limit — return original section
    if available_len <= 0:
        logger.warning(f"Section title too long to split: {title[:20]}...")
        return [section]

    # Strip duplicate title from body start to avoid redundant content in sub-chunks
    body = content
    if title and body.lstrip().startswith(title):
        body = body[body.find(title) + len(title):].lstrip()

    # Initialise LangChain recursive splitter with priority-ordered separators
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=available_len,  # Max body length (title already excluded)
        chunk_overlap=0,           # No overlap: heading-based sections are semantically complete
        # Separator priority: blank line → newline → CJK punctuation → EN punctuation → space
        separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],
    )

    # Split body and assemble sub-sections with full metadata for traceability
    sub_sections = []
    for idx, chunk in enumerate(splitter.split_text(body), start=1):
        text = chunk.strip()
        if not text:
            continue
        full_text = (prefix + text).strip()
        sub_sections.append(
            {
                "title": f"{title}-{idx}" if title else f"chunk-{idx}",  # Sub-chunk title with index
                "content": full_text,                    # Full content including title prefix
                "parent_title": title,                   # Parent section title for merging/tracing
                "part": idx,                             # Sub-chunk index
                "file_title": section.get("file_title"), # Owning file title
            }
        )

    logger.debug(f"Long section split: {title} → {len(sub_sections)} sub-chunk(s)")
    return sub_sections


def _merge_short_sections(
    sections: List[Dict[str, Any]], min_length: int = MIN_CONTENT_LENGTH
) -> List[Dict[str, Any]]:
    """
    Helper: merge undersized adjacent chunks to reduce fragmentation and improve retrieval quality.
    Merge rule: only merge chunks that share the same parent_title AND are below the min length threshold.
    Cross-section merging is intentionally avoided.
    :param sections: List of chunks to process (typically output of _split_long_section)
    :param min_length: Minimum length threshold; chunks below this are candidates for merging
    :return: Merged chunk list with preserved metadata
    """
    # Edge case: empty list — return immediately
    if not sections:
        logger.debug("Chunk list is empty, returning as-is")
        return []

    merged_sections = []
    current_chunk = None  # Accumulator: holds the chunk currently being built up

    for sec in sections:
        # Initialise: first chunk becomes the current accumulator
        if current_chunk is None:
            current_chunk = sec
            continue

        # Merge conditions: 1) current chunk is below min length, 2) same parent title
        is_current_short = len(current_chunk["content"]) < min_length
        is_same_parent = current_chunk.get("parent_title") == sec.get("parent_title")

        if is_current_short and is_same_parent:
            # Strip duplicate parent title from the start of the next chunk before merging
            parent_title = sec.get("parent_title", "")
            next_content = sec["content"]
            if parent_title and next_content.startswith(parent_title):
                next_content = next_content[len(parent_title):].lstrip()
            # Merge with blank line separator for clean formatting
            current_chunk["content"] += "\n\n" + next_content
            # Update part index to the latest for traceability
            if "part" in sec:
                current_chunk["part"] = sec["part"]
            logger.debug(
                f"Merging short chunk: {current_chunk.get('parent_title')} → cumulative length {len(current_chunk['content'])}"
            )
        else:
            # Merge conditions not met: commit current chunk and start fresh
            merged_sections.append(current_chunk)
            current_chunk = sec

    # Commit the final accumulated chunk
    if current_chunk is not None:
        merged_sections.append(current_chunk)

    logger.debug(f"Short chunk merge complete: {len(sections)} → {len(merged_sections)} chunk(s)")
    return merged_sections


def step_4_refine_chunks(
    sections: List[Dict[str, Any]], max_len: int
) -> List[Dict[str, Any]]:
    """
    Step 4: Chunk refinement — split long sections, merge short ones.
    Phase 1: Split oversized sections so all chunks are within max_len.
    Phase 2: Merge undersized adjacent chunks to reduce fragmentation.
    Phase 3: Ensure parent_title is populated on all chunks (required by Milvus schema).
    :param sections: Section list from step 3
    :param max_len: Max character length per chunk
    :return: Refined chunk list — appropriately sized, low fragmentation, schema-compliant
    """
    # Guard: invalid max length — skip refinement and return as-is
    if not max_len or max_len <= 0:
        logger.warning(f"Step 4: Invalid max chunk length ({max_len}), skipping refinement")
        return sections

    # Phase 1: Split oversized sections
    refined_split = []
    for sec in sections:
        # extend() flattens the returned list directly into refined_split (avoids nesting)
        refined_split.extend(_split_long_section(sec, max_len))
    logger.info(f"Step 4-1: Long section split complete — {len(refined_split)} initial sub-chunk(s)")

    # Phase 2: Merge undersized chunks
    final_sections = _merge_short_sections(refined_split)
    logger.info(f"Step 4-2: Short chunk merge complete — {len(final_sections)} final chunk(s)")

    # Phase 3: Ensure parent_title and part fields are present on all chunks (Milvus schema requirement)
    for sec in final_sections:
        if not isinstance(sec, dict):
            continue
        if "part" not in sec:
            sec["part"] = 0
        if not sec.get("parent_title"):
            sec["parent_title"] = sec.get("title") or ""
    logger.debug("Step 4-3: parent_title backfill complete — all chunks are schema-compliant")

    return final_sections


def step_5_print_stats(lines_count: int, sections: List[Dict[str, Any]]) -> None:
    """
    Step 5: Log document splitting statistics for monitoring and debugging.
    :param lines_count: Total line count of the raw MD content
    :param sections: Final processed chunk list
    """
    chunk_num = len(sections)
    logger.info("-" * 50 + " Document Split Statistics " + "-" * 50)
    logger.info(f"Raw MD line count: {lines_count}")
    logger.info(f"Total chunks generated: {chunk_num}")
    if sections:
        first_title = sections[0].get("title", "Untitled")
        logger.info(f"First chunk title preview: {first_title}")
    logger.info("-" * 110)


def step_6_backup(state: ImportGraphState, sections: List[Dict[str, Any]]) -> None:
    """
    Step 6: Back up chunk results to a local JSON file for debugging and reuse.
    :param state: Pipeline state dict; must contain 'local_dir' for the backup path
    :param sections: Final processed chunk list
    """
    local_dir = state.get("local_dir")
    if not local_dir:
        logger.warning("Step 6: No backup directory configured (local_dir), skipping chunk backup")
        return

    try:
        os.makedirs(local_dir, exist_ok=True)
        backup_path = os.path.join(local_dir, "chunks.json")
        with open(backup_path, "w", encoding="utf-8") as f:
            # json.dump serialises the nested Python structure (List[Dict]) directly to a JSON file.
            # ensure_ascii=False preserves non-ASCII characters (e.g. Chinese) as readable text
            # instead of \uXXXX escape sequences.
            json.dump(
                sections,
                f,
                ensure_ascii=False,  # Preserve Unicode characters for human readability
                indent=2,
            )
        logger.info(f"Step 6: Chunk backup saved — path: {backup_path}")
    except Exception as e:
        # Backup failure is non-fatal: log the error and continue
        logger.error(f"Step 6: Chunk backup failed — error: {str(e)}", exc_info=False)


def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    Node: Document Splitting (node_document_split)
    Split long documents into smaller Chunks (slices) for easier retrieval.
    Implementations:
    1. Recursively split based on Markdown heading levels.
    2. Perform secondary splitting on overly long paragraphs.
    3. Generate a list of Chunks containing Metadata (heading paths).
    """
    node_name = sys._getframe().f_code.co_name
    logger.info(f">>> Starting node: [Document Split] {node_name}")

    add_running_task(state["task_id"], node_name)

    try:
        # Step 1: Load and normalise input data
        # Extracts MD content, file title, and max chunk length; normalises line endings;
        # terminates early if no valid MD content is present.
        content, file_title, max_len = step_1_get_inputs(state)
        if content is None:
            logger.info(f">>> Node terminated early: {node_name} (no valid MD content)")
            return state

        # Step 2: Initial split by MD headings
        # Splits the document into sections by heading level (#/##/###),
        # skipping headings inside code blocks to preserve semantic integrity.
        sections, title_count, lines_count = step_2_split_by_titles(content, file_title)

        # Step 3: Fallback handling for no-heading documents
        # Wraps the full content as a single untitled section if no headings were found,
        # ensuring a consistent data format for downstream steps.
        sections = step_3_handle_no_title(content, sections, title_count, file_title)

        # Step 4: Chunk refinement (split long, merge short)
        # Splits oversized sections (paragraph → sentence), then merges undersized adjacent chunks
        # with the same parent title to reduce fragmentation.
        # Also backfills parent_title on all chunks for Milvus schema compliance.
        sections = step_4_refine_chunks(sections, max_len)

        # Step 5: Log splitting statistics
        # Outputs raw line count, final chunk count, and first chunk title for monitoring.
        step_5_print_stats(lines_count, sections)

        # Step 6: Backup chunk results and update state
        # Writes chunk list to chunks.json in local_dir (skipped if local_dir is not set).
        # Updates state["chunks"] for downstream nodes (e.g. vector store ingestion).
        state["chunks"] = sections
        step_6_backup(state, sections)

        logger.info(
            f">>> Node complete: [Document Split] {node_name} — {len(sections)} chunk(s) written to state"
        )

    except Exception as e:
        # Global exception handler: log the error without crashing the pipeline
        logger.error(
            f">>> Node failed: [Document Split] {node_name} — error: {str(e)}",
            exc_info=True,
        )

    return state


if __name__ == '__main__':
    """
    Integration test: runs node_md_img followed by node_document_split.
    Prerequisites: 1) .env configured (MinIO / LLM), 2) test MD file exists, 3) node_md_img importable.
    Flow: image processing → document splitting → validate end-to-end output.
    """
    from app.utils.path_util import PROJECT_ROOT
    from app.import_process.agent.nodes.node_md_img import node_md_img

    logger.info(f"Local test - project root: {PROJECT_ROOT}")

    test_md_name = os.path.join(
        "output",
        "The_IoT_and_AI_in_Agriculture_The_Time_Is_Now_A_Systematic_Review_of_Smart_Sensing_Technologies",
        "The_IoT_and_AI_in_Agriculture_The_Time_Is_Now_A_Systematic_Review_of_Smart_Sensing_Technologies.md"
    )
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    if not os.path.exists(test_md_path):
        logger.error(f"Local test - test file not found: {test_md_path}")
        logger.info("Please check the file path or place a valid MD file in the output directory")
    else:
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": "",
            "file_title": "The_IoT_and_AI_in_Agriculture_The_Time_Is_Now_A_Systematic_Review_of_Smart_Sensing_Technologies",
            "local_dir": os.path.join(PROJECT_ROOT, "output"),
        }
        logger.info("Starting local test - MD image processing pipeline")
        result_state = node_md_img(test_state)
        logger.info(f"Image processing complete - state: result_state")
        logger.info("\n=== Starting document split node integration test ===")

        logger.info(">> Running node: node_document_split")
        final_state = node_document_split(result_state)
        final_chunks = final_state.get("chunks", [])
        logger.info(f"Test complete: {len(final_chunks)} chunk(s) generated")
