from pathlib import Path
import subprocess
import sys
import time


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    commands = [
        [sys.executable, str(root / "scripts" / "serve_chat.py")],
        [sys.executable, str(root / "scripts" / "serve_image.py")],
    ]
    processes = []
    try:
        for command in commands:
            processes.append(subprocess.Popen(command, cwd=root))
        print("ChatV1 presentation services started.")
        print("Chat API:  http://127.0.0.1:8001")
        print("Image API: http://127.0.0.1:8000")
        print("Open web/chatv1-demo.html in your browser.")
        while all(p.poll() is None for p in processes):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()


if __name__ == "__main__":
    main()
