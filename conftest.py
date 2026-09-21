"""Put code/ on the import path so the tests can import the modules directly."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code"))
