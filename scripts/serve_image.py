import argparse
from http.server import HTTPServer

from chatv1.diffusion import DiffusionConfig, LocalDiffusionGenerator
from chatv1.http import make_image_handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local ChatV1 image API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    args = parser.parse_args()

    generator = LocalDiffusionGenerator(DiffusionConfig(model_id=args.model))
    server = HTTPServer((args.host, args.port), make_image_handler(lambda: generator))
    print(f"ChatV1 image API: http://{args.host}:{args.port}/v1/images/generate")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
