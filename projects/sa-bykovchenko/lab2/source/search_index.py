from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from search_cli import main  # noqa: E402


if __name__ == '__main__':
    sys.exit(main(default_source='index', default_backend='flat_ip'))
