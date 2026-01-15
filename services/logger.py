import logging

logging.basicConfig(
    filename="genai.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("genai-app")

