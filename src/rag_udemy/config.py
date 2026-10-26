import os

# Flag to enable/disable answer verification
USE_VERIFIER: bool = os.getenv("USE_VERIFIER", "true").lower() in ("true", "1", "yes")
