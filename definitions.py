"""This file contains global definitions"""

import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_DIR = os.path.join(ROOT_DIR, "logs")
SAVED_AGENTS = os.path.join(ROOT_DIR, "saved_agents")