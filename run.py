import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from pl_mai_vps.task import moment_retrieval

moment_retrieval.main()
