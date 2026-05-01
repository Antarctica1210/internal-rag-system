import json

from dotenv import load_dotenv
import sys
from pathlib import Path

# Add parent directory to path so imports work when running this test directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.import_process.agent.main_graph import get_kb_import_workflow
from app.import_process.agent.state import create_default_state
from app.core.logger import logger

load_dotenv()

logger.info("===== start testing graph =====")

initial_state = create_default_state(local_file_path="RS-12-guidance.md")
final_state = None

kb_import_app = get_kb_import_workflow()

for event in kb_import_app.stream(initial_state):
    for key, value in event.items():
        logger.info(f"node: {key}")
        final_state = value

logger.info(f"Final State: {json.dumps(final_state, indent=4, ensure_ascii=False)}")

logger.info("Graph Structure:")
# print the graph structure in ASCII format for visualization
kb_import_app.get_graph().print_ascii()

logger.info("===== Testing Complete =====")