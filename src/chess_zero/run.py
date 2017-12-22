import os
import sys

_PATH_ = os.path.dirname(os.path.dirname(__file__))


if _PATH_ not in sys.path:
	sys.path.append(_PATH_)


if __name__ == "__main__":
	sys.setrecursionlimit(10000)
	from chess_zero import manager
	manager.start()