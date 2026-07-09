import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.screen_caption_boundaries import main


if __name__ == "__main__":
    main()
