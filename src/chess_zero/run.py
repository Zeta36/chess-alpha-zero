
import os
import sys
from dotenv import Dotenv
dotenv = Dotenv(".env") # Of course, replace by your correct path

os.environ.update(dotenv)

_PATH_ = os.path.dirname(os.path.dirname(__file__))


if _PATH_ not in sys.path:
    sys.path.append(_PATH_)


if __name__ == "__main__":
    from chess_zero import manager
    manager.start()
