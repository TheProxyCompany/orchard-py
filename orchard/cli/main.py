import argparse
import asyncio
import logging
import sys
from pathlib import Path

import uvicorn

from orchard.engine.fetch import download_engine, get_installed_version
from orchard.engine.inference_engine import InferenceEngine
from orchard.server.app import create_app

# Configure logging for the CLI
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s"
)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


async def run_serve(args: argparse.Namespace):
    """Handler for the 'serve' command."""
    logger.debug("Initializing local inference engine...")
    engine_log_path = Path(args.engine_log_file) if args.engine_log_file else None
    async with InferenceEngine(
        engine_log_file=engine_log_path,
        startup_timeout=args.engine_startup_timeout,
    ) as local_engine:
        logger.debug("Engine started.")
        if args.models:
            await local_engine.load_models(args.models)

        fastapi_app = create_app(local_engine)
        config = uvicorn.Config(
            fastapi_app,
            host=args.host,
            port=args.port,
            log_level="warning",
            loop="asyncio",
        )
        server = uvicorn.Server(config)
        await server.serve()

    logger.debug("Server has shut down.")


def run_engine_stop(args: argparse.Namespace):
    """Handler for the 'engine stop' command."""
    logger.info("Attempting to shut down the shared orchard engine process...")
    if InferenceEngine.shutdown(timeout=args.timeout):
        logger.info("Engine shutdown successful.")
    else:
        logger.warning("Engine shutdown was forceful or timed out.")
        sys.exit(1)


def run_upgrade(args: argparse.Namespace):
    """Handler for the 'upgrade' command."""
    channel = args.channel
    current = get_installed_version()

    if current:
        print(f"\033[34m→\033[0m Current version: {current}")
    else:
        print("\033[34m→\033[0m No version currently installed")

    print(f"\033[34m→\033[0m Fetching latest from '{channel}' channel...")

    try:
        download_engine(channel=channel)
        new_version = get_installed_version()
        if new_version == current:
            print(f"\033[32m✓\033[0m Already on latest: {new_version}")
        else:
            print(f"\033[32m✓\033[0m Upgraded to {new_version}")
    except Exception as e:
        print(f"\033[31m✗\033[0m Upgrade failed: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point for the Proxy Inference Engine."""
    parser = argparse.ArgumentParser(
        description="Proxy Inference Engine (PIE) command-line interface."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- 'serve' command ---
    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI server.")
    serve_parser.add_argument(
        "--engine-log-file", type=str, default=None, help="Path to the engine log file."
    )
    serve_parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host to bind the server to."
    )
    serve_parser.add_argument(
        "--port", type=int, default=8000, help="Port to listen on."
    )
    serve_parser.add_argument(
        "--model",
        "--models",
        dest="models",
        nargs="*",
        help="Model identifier(s) to preload on startup.",
    )
    serve_parser.add_argument(
        "--engine-startup-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for the C++ engine to start.",
    )
    serve_parser.set_defaults(func=run_serve)

    # --- 'upgrade' command ---
    upgrade_parser = subparsers.add_parser(
        "upgrade", help="Download and install the latest engine binary."
    )
    upgrade_parser.add_argument(
        "channel",
        nargs="?",
        default="stable",
        help="Release channel to pull from (default: stable).",
    )
    upgrade_parser.set_defaults(func=run_upgrade)

    # --- 'engine' command group ---
    engine_parser = subparsers.add_parser(
        "engine", help="Manage the background engine process."
    )
    engine_subparsers = engine_parser.add_subparsers(
        dest="engine_command", required=True
    )

    # --- 'engine stop' command ---
    stop_parser = engine_subparsers.add_parser(
        "stop", help="Stop the shared orchard engine process."
    )
    stop_parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Seconds to wait for graceful shutdown.",
    )
    stop_parser.set_defaults(func=run_engine_stop)

    args = parser.parse_args()
    try:
        if asyncio.iscoroutinefunction(args.func):
            asyncio.run(args.func(args))
        else:
            args.func(args)
    except KeyboardInterrupt:
        logger.debug("Keyboard interrupt received.")
        sys.exit(0)
    except Exception as e:
        logger.error("Error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
