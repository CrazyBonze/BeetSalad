# import os
# from pathlib import Path
# 
# 
# def main() -> None:
#     project_root = Path(__file__).resolve().parent
#     beetsdir = project_root / "config"
# 
#     # Make the whole project self-contained: config + library db + state files.
#     os.environ.setdefault("BEETSDIR", str(beetsdir))
# 
#     # Import AFTER env var is set
#     from beets.ui import main as beet_main
# 
#     # Just forward args to beets; no need to inject -c if you use BEETSDIR.
#     beet_main()
# 
# 
# if __name__ == "__main__":
#     main()
# 