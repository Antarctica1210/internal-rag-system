import os
import re
import sys
import base64
from pathlib import Path
from typing import Dict, List, Tuple
from collections import deque

# MinIO dependencies
from minio import Minio
from minio.deleteobjects import DeleteObject

# Remove native OpenAI; use LangChain utility class and multimodal message module instead
from app.clients.minio_utils import get_minio_client
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task

# LLM client utility (core reuse, replaces native OpenAI calls)
from app.lm.lm_utils import get_llm_client

# LangChain multimodal dependencies (message construction + exception handling)
from langchain.messages import HumanMessage
from langchain_core.exceptions import LangChainException

# Project config
from app.conf.minio_config import minio_config
from app.conf.lm_config import lm_config

# Project logger
from app.core.logger import logger

# API rate limiting utility
from app.utils.rate_limit_utils import apply_api_rate_limit

# Prompt loading utility
from app.core.load_prompt import load_prompt

# Supported image formats for MinIO (lowercase extensions, unified matching standard)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

"""Step 1: Initialise core MD data — get content, file path, and images folder path"""


def step_1_get_content(state: ImportGraphState) -> Tuple[str, Path, Path]:
    """
    Extract and initialise the core data required for MD processing from the global state.
    :param state: Global state object for the import pipeline
    :return: Tuple of (MD file content, MD file Path object, images folder Path object)
    :raise FileNotFoundError: When no valid MD file path exists in the state
    """
    md_file_path = state["md_path"]
    # Validate MD file path
    if not md_file_path:
        raise FileNotFoundError(f"No valid MD file path in global state: {state['md_path']}")

    path_obj = Path(md_file_path)
    # Use MD content from state if available; otherwise read from file
    if not state["md_content"]:
        with open(path_obj, "r", encoding="utf-8") as f:
            md_content = f.read()
        logger.debug(f"MD content loaded from file, size: {len(md_content)} chars")
    else:
        md_content = state["md_content"]
        logger.debug(f"MD content loaded from global state, size: {len(md_content)} chars")

    # Images folder is always the 'images' directory alongside the MD file
    images_dir = path_obj.parent / "images"
    return md_content, path_obj, images_dir


"""Step 2"""


def is_supported_image(filename: str) -> bool:
    """
    Check whether a file is a MinIO-supported image format (case-insensitive extension match).
    :param filename: Filename including extension
    :return: True if supported, False otherwise
    """
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def find_image_in_md(
    md_content: str, image_filename: str, context_len: int = 100
) -> List[Tuple[str, str]]:
    """
    Find all references to a given image in the MD content and return surrounding context.
    :param md_content: Full MD file content
    :param image_filename: Image filename including extension
    :param context_len: Number of characters to capture before and after each match (default 100)
    :return: List of (pre_text, post_text) tuples; empty list if no matches found
    """
    # Escape special characters in the filename to avoid regex syntax errors;
    # compile the pattern for efficiency.
    # 'r' prefix = raw string: tells Python not to process escape sequences (e.g. \, \n, \t).
    pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_filename) + r".*?\)")
    results = []

    # Iterate over all MD image tag matches
    for m in pattern.finditer(md_content):
        start, end = m.span()
        # Extract surrounding context, guarding against index out-of-bounds
        pre_text = md_content[max(0, start - context_len) : start]
        post_text = md_content[end : min(len(md_content), end + context_len)]
        logger.debug(f"Image [{image_filename}] reference found, pre-text: {pre_text.strip()}")
        logger.debug(f"Image [{image_filename}] reference found, post-text: {post_text.strip()}")
        results.append((pre_text, post_text))

    if not results:
        logger.debug(f"No reference to image [{image_filename}] found in MD content")
    return results


# Step 2: Scan the images folder and filter for images actually referenced in the MD
def step_2_scan_images(
    md_content: str, images_dir: Path
) -> List[Tuple[str, str, Tuple[str, str]]]:
    """
    Scan the images folder and filter for supported-format images that are referenced in the MD.
    Assembles processing metadata for each qualifying image.
    :param md_content: Full MD file content
    :param images_dir: Path object for the images folder
    :return: List of (image_filename, full_image_path, context) tuples for images to process
    """
    targets = []
    # Iterate over all files in the images folder
    for image_file in os.listdir(images_dir):
        # Skip unsupported image formats
        if not is_supported_image(image_file):
            logger.debug(f"Unsupported image format, skipping: {image_file}")
            continue

        # Build full image path
        img_path = str(images_dir / image_file)
        # Find the image's reference context in the MD
        context_list = find_image_in_md(md_content, image_file)

        # Skip images not referenced in the MD
        if not context_list:
            logger.warning(f"Image not referenced in MD, skipping: {image_file}")
            continue

        # Use the first matched context
        targets.append((image_file, img_path, context_list[0]))
        logger.info(f"Image added to processing list: {image_file}")

    logger.info(f"Image scan complete, {len(targets)} image(s) queued for processing")
    return targets


"""Step 3"""


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode a local image file as a Base64 string (for multimodal LLM input).
    :param image_path: Full local path to the image file
    :return: Base64-encoded string of the image (UTF-8 decoded)
    """
    with open(image_path, "rb") as img_file:
        base64_str = base64.b64encode(img_file.read()).decode("utf-8")
    logger.debug(f"Image Base64 encoding complete, file: {image_path}, encoded length: {len(base64_str)}")
    return base64_str


def summarize_image(
    image_path: str, root_folder: str, image_content: Tuple[str, str]
) -> str:
    """
    Call a multimodal LLM to generate a content summary for an image.
    Uses the project's unified LangChain LLM client. The summary is used as the
    Markdown image alt-text, strictly limited to 50 characters.
    :param image_path: Full local path to the image file
    :param root_folder: Document folder/stem name, providing context to the LLM
    :param image_content: Context tuple from MD — (pre_text, post_text)
    :return: Image content summary; returns "image description" as fallback on error
    """
    # Encode image as Base64 for multimodal LLM input
    base64_image = encode_image_to_base64(image_path)
    try:
        # 1. Get the project's unified LLM client (auto-cached, using the vision model name)
        lvm_client = get_llm_client(model=lm_config.lv_model)

        # Load and render the prompt (pass all placeholder variables)
        prompt_text = load_prompt(
            name="image_summary",        # Prompt filename (without .prompt extension)
            root_folder=root_folder,     # Maps to {root_folder}
            image_content=image_content, # Maps to {image_content[0]} and {image_content[1]}
        )

        # 2. Build a LangChain standard multimodal HumanMessage (compatible with Qwen/OpenAI vision models)
        messages = [
            HumanMessage(
                content=[
                    # Text prompt: carries context and defines summary rules
                    {"type": "text", "text": prompt_text},
                    # Multimodal core: Base64-encoded image data
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ]
            )
        ]

        # 3. Standard LangChain invocation via invoke() (timeout/retry handled by the utility class)
        response = lvm_client.invoke(messages)

        # 4. Parse response (LangChain always returns a unified 'content' field — no multi-level parsing needed)
        summary = response.content.strip().replace("\n", "")
        logger.info(f"Image summary generated successfully: {image_path}, summary: {summary}")
        return summary

    except LangChainException as e:
        logger.error(f"Image summary failed (LangChain exception): {image_path}, error: {str(e)}")
        return "image description"
    except Exception as e:
        logger.error(f"Image summary failed (system exception): {image_path}, error: {str(e)}")
        return "image description"


def step_3_generate_summaries(
    doc_stem: str,
    targets: List[Tuple[str, str, Tuple[str, str]]],
    requests_per_minute: int = 9,
) -> Dict[str, str]:
    """
    Step 3: Batch-generate content summaries for all queued images, with API rate limiting
    to avoid triggering LLM throttling.
    :param doc_stem: Document filename without extension, used as LLM prompt context
    :param targets: List of (image_filename, full_image_path, context) tuples
    :param requests_per_minute: Max API requests per minute (default 9 — adjust per LLM limits)
    :return: Dict of image summaries: key = image filename, value = content summary
    """
    summaries = {}
    request_times = deque()  # Initialise request time queue outside loop for reuse across iterations

    for img_file, image_path, context in targets:
        apply_api_rate_limit(request_times, requests_per_minute, window_seconds=60)
        logger.debug(f"Generating image summary: {image_path}")
        summaries[img_file] = summarize_image(
            image_path, root_folder=doc_stem, image_content=context
        )

    logger.info(f"Batch image summary generation complete, processed {len(summaries)} image(s)")
    return summaries


"""Step 4"""


def step_4_upload_and_replace(
    minio_client: Minio,
    doc_stem: str,
    targets: List[Tuple[str, str, Tuple[str, str]]],
    summaries: Dict[str, str],
    md_content: str,
) -> str:
    """
    Step 4: Core pipeline — upload images to MinIO, merge summaries & URLs, replace MD image references.
    Full flow: clean old MinIO directory → batch upload new images → merge summaries and URLs → replace MD content.
    :param minio_client: Initialised MinIO client object
    :param doc_stem: Document filename without extension, used as MinIO subdirectory name (per-document isolation)
    :param targets: List of (image_filename, full_image_path, context) tuples
    :param summaries: Dict of image summaries: key = image filename, value = content summary
    :param md_content: Original MD file content
    :return: Updated MD content with image references replaced
    """
    # Build MinIO upload directory: config root dir + document stem (spaces removed to avoid path issues)
    minio_img_dir = minio_config.minio_img_dir
    upload_dir = f"{minio_img_dir}/{doc_stem}".replace(" ", "")

    # Sub-step 1: Clean the old MinIO directory for this document (ensures idempotency)
    clean_minio_directory(minio_client, upload_dir)
    # Sub-step 2: Batch-upload images to MinIO and get URL mapping
    urls = upload_images_batch(minio_client, upload_dir, targets)
    # Sub-step 3: Merge image summaries and URLs, filtering out failed uploads
    image_info = merge_summary_and_url(summaries, urls)
    # Sub-step 4: Replace local image references in MD with MinIO remote references
    if image_info:
        md_content = process_md_file(md_content, image_info)

    return md_content


def clean_minio_directory(minio_client: Minio, prefix: str) -> None:
    """
    Idempotently delete all existing files under a MinIO directory prefix,
    preventing stale/duplicate file accumulation. Safe to call when no files exist.
    :param minio_client: Initialised MinIO client object
    :param prefix: MinIO directory prefix to clean
    """
    try:
        # List all objects under the given prefix (recursive)
        objects_to_delete = minio_client.list_objects(
            bucket_name=minio_config.bucket_name, prefix=prefix, recursive=True
        )
        # Build deletion list
        delete_list = [DeleteObject(obj.object_name) for obj in objects_to_delete]

        if delete_list:
            logger.info(f"Cleaning MinIO directory, {len(delete_list)} file(s) to delete, prefix: {prefix}")
            # Batch delete objects
            errors = minio_client.remove_objects(minio_config.bucket_name, delete_list)
            for error in errors:
                logger.error(f"MinIO file deletion error: {error}")
        else:
            logger.debug(f"MinIO directory is already empty, nothing to clean: {prefix}")
    except Exception as e:
        logger.error(f"MinIO directory cleanup failed: {prefix}, error: {str(e)}")


def upload_images_batch(
    minio_client: Minio,
    upload_dir: str,
    targets: List[Tuple[str, str, Tuple[str, str]]],
) -> Dict[str, str]:
    """
    Batch-upload queued images to MinIO and return a filename-to-URL mapping.
    :param minio_client: Initialised MinIO client object
    :param upload_dir: MinIO upload root directory
    :param targets: List of (image_filename, full_image_path, context) tuples
    :return: Dict of image URLs: key = image filename, value = MinIO access URL
    """
    urls = {}
    for img_file, img_path, _ in targets:
        # Build MinIO object name
        object_name = f"{upload_dir}/{img_file}"
        logger.debug(f"MinIO object name constructed: {object_name}")
        # Upload single image and capture URL.
        # ':=' is the walrus operator (Python 3.8+): assigns and evaluates in a single expression,
        # replacing the traditional assign-then-check two-liner.
        if img_url := upload_to_minio(minio_client, img_path, object_name):
            urls[img_file] = img_url
    logger.info(f"Batch upload complete, {len(urls)}/{len(targets)} image(s) uploaded successfully")
    return urls


def upload_to_minio(
    minio_client: Minio, local_path: str, object_name: str
) -> str | None:
    """
    Upload a single local image to MinIO object storage and return its publicly accessible URL.
    :param minio_client: Initialised MinIO client object
    :param local_path: Full local path to the image file
    :param object_name: Destination object name in MinIO (including directory path)
    :return: MinIO access URL for the image, or None if upload fails
    """
    try:
        logger.info(f"Uploading image to MinIO: local_path={local_path}, object_name={object_name}")
        # Upload local file to MinIO (fput_object: file-stream upload, suitable for large files)
        minio_client.fput_object(
            bucket_name=minio_config.bucket_name,  # MinIO bucket name (from config)
            object_name=object_name,               # MinIO object name
            file_path=local_path,                  # Local file path
            # Auto-infer image Content-Type (e.g. image/png, image/jpeg).
            # os.path.splitext returns (root, ext) where ext includes the leading dot (e.g. '.jpg').
            # [1:] strips the dot to get the bare extension for the MIME type.
            content_type=f"image/{os.path.splitext(local_path)[1][1:]}",
        )

        # Escape backslashes in object name to avoid URL parsing errors
        object_name = object_name.replace("\\", "%5C")
        # Select HTTP or HTTPS based on config
        protocol = "https" if minio_config.minio_secure else "http"
        # Build MinIO base access URL
        base_url = f"{protocol}://{minio_config.endpoint}/{minio_config.bucket_name}"
        # Concatenate full image URL (base_url already ends without trailing slash)
        img_url = f"{base_url}{object_name}"
        logger.info(f"Image uploaded successfully, URL: {img_url}")
        return img_url
    except Exception as e:
        logger.error(f"MinIO upload failed: {local_path}, error: {str(e)}")
        return None


def merge_summary_and_url(
    summaries: Dict[str, str], urls: Dict[str, str]
) -> Dict[str, Tuple[str, str]]:
    """
    Merge the image summaries dict and URLs dict, filtering out images that failed to upload.
    :param summaries: Dict of summaries: key = image filename, value = content summary
    :param urls: Dict of URLs: key = image filename, value = MinIO access URL
    :return: Merged dict: key = image filename, value = (summary, URL) tuple
    """
    image_info = {}
    # Keep only images that have a corresponding URL (i.e. upload succeeded)
    for image_file, summary in summaries.items():
        if url := urls.get(image_file):
            image_info[image_file] = (summary, url)
    logger.info(f"Summary and URL merge complete, {len(image_info)} valid image record(s)")
    return image_info


def process_md_file(md_content: str, image_info: Dict[str, Tuple[str, str]]) -> str:
    """
    Replace local image references in MD content with MinIO remote references.
    Replacement rule: ![original alt](local path) → ![image summary](MinIO URL)
    :param md_content: Original MD file content
    :param image_info: Merged image info dict: key = image filename, value = (summary, URL)
    :return: Updated MD content with replaced image references
    """
    for img_filename, (summary, new_url) in image_info.items():
        # Regex matches MD image tags case-insensitively, compatible with varied path formats.
        # Pattern: ![any alt text](any path + image filename + any suffix)
        pattern = re.compile(
            r"!\[.*?\]\(.*?" + re.escape(img_filename) + r".*?\)", re.IGNORECASE
        )
        # Replace matched content: new summary as alt text, new URL as image path.
        # Note: if summary or new_url could ever contain backslashes, prefer the lambda form:
        # md_content = pattern.sub(lambda m: f"![{summary}]({new_url})", md_content)
        md_content = pattern.sub(f"![{summary}]({new_url})", md_content)
        logger.debug(f"MD image reference replaced: {img_filename} → {new_url}")

    logger.info(f"MD image reference replacement complete, {len(image_info)} reference(s) replaced")
    logger.debug(
        f"Updated MD content preview: {md_content[:500]}..."
        if len(md_content) > 500
        else f"Updated MD content: {md_content}"
    )
    return md_content


"""Step 5: Save processed MD file as a new file"""


def step_5_backup_new_md_file(origin_md_path: str, md_content: str) -> str:
    """
    Step 5: Save processed MD content to a new file, leaving the original unchanged.
    Naming convention: original filename + _new.md (e.g. test.md → test_new.md)
    :param origin_md_path: Full path to the original MD file
    :param md_content: Processed MD content to save
    :return: Full path to the new MD file
    """
    # Build new file path by replacing the original extension with _new.md
    new_md_file_name = os.path.splitext(origin_md_path)[0] + "_new.md"

    # Write new MD content (overwrites if file already exists)
    with open(new_md_file_name, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info(f"Processed MD file saved, new file path: {new_md_file_name}")
    return new_md_file_name


def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    Node: Image Processing (node_md_img)
    Why this name: Process image resources (Image) in Markdown.
    Future implementations:
    1. Scan image links in Markdown.
    2. Upload images to MinIO object storage.
    3. (Optional) Call multimodal model to generate image descriptions.
    4. Replace image links in Markdown with MinIO URLs.
    """

    # Record current running task for monitoring and debugging
    add_running_task(state["task_id"], sys._getframe().f_code.co_name)

    # Step 1: get core data from MD file
    md_content, path_obj, images_dir = step_1_get_content(state)
    state["md_content"] = md_content

    # No images folder — skip all image processing
    if not images_dir.exists():
        logger.info(f"Images folder does not exist, skipping image processing: {images_dir.absolute()}")
        return state

    # Initialise MinIO client; skip image processing if it fails
    minio_client = get_minio_client()
    if not minio_client:
        logger.warning("MinIO client initialisation failed, skipping image processing pipeline")
        return state

    # Step 2: scan and filter supported images referenced in the MD
    # Returns: (image_file, img_path, context_list[0])
    targets = step_2_scan_images(md_content, images_dir)
    if not targets:
        logger.info("No supported images referenced in MD, skipping further processing")
        return state

    # Step 3: call multimodal LLM to generate image summaries
    summaries = step_3_generate_summaries(path_obj.stem, targets)

    # Step 4: upload images to MinIO, replace MD image paths with URLs and inject summaries
    new_md_content = step_4_upload_and_replace(
        minio_client, path_obj.stem, targets, summaries, md_content
    )
    state["md_content"] = new_md_content

    # Step 5: save new MD file and update file path in state
    new_md_file_name = step_5_backup_new_md_file(state['md_path'], new_md_content)
    state["md_path"] = new_md_file_name
    logger.info(f"MD image processing complete, new file saved: {new_md_file_name}")

    return state


if __name__ == "__main__":
    from app.utils.path_util import PROJECT_ROOT
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
            "md_content": ""
        }
        logger.info("Starting local test - MD image processing pipeline")
        result_state = node_md_img(test_state)
        logger.info(f"Local test complete - result state: {result_state}")