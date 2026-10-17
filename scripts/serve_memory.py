import argparse
import os

import uvicorn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        default=os.getenv("AMC_HOST", "127.0.0.1"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("AMC_PORT", "8000")),
    )
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "amc.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
