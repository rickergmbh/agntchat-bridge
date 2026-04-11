"""AgentChat CLI — python -m agentchat <command>

Commands:
    join <code>   Claim an invite, save credentials, start executor
    info <code>   Show public invite info
    status        Show saved credentials
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("agentchat.cli")

DEFAULT_GATEWAY_URL = "https://agentchat-backend.fly.dev"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m agentchat",
        description="AgentChat CLI — claim invites and manage agents",
    )
    subparsers = parser.add_subparsers(dest="command")

    # join
    join_parser = subparsers.add_parser("join", help="Claim invite and start executor")
    join_parser.add_argument("code", help="Invite code (e.g. inv_Abc123)")
    join_parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL, help="Backend URL")
    join_parser.add_argument("--executor-key", default=None, help="Executor key for gateway registration")
    join_parser.add_argument("--display-name", default=None, help="Display name for the executor")
    join_parser.add_argument("--capabilities", default=None, help="Comma-separated capabilities (e.g. code,git,shell)")
    join_parser.add_argument("--no-start", action="store_true", help="Don't start the executor poll loop")

    # info
    info_parser = subparsers.add_parser("info", help="Show public invite info")
    info_parser.add_argument("code", help="Invite code")
    info_parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL, help="Backend URL")

    # status
    subparsers.add_parser("status", help="Show saved credentials")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "join":
        asyncio.run(_cmd_join(args))
    elif args.command == "info":
        asyncio.run(_cmd_info(args))
    elif args.command == "status":
        _cmd_status()


async def _cmd_join(args: argparse.Namespace) -> None:
    from .invite import claim_invite, save_credentials
    from .executor import ExecutorClient, GatewayTask

    executor_key = args.executor_key or _default_executor_key()
    executor_display_name = args.display_name or f"CLI ({executor_key})"
    claim_capabilities = args.capabilities.split(",") if args.capabilities else None

    logger.info("Claiming invite %s...", args.code)

    try:
        result = await claim_invite(
            gateway_url=args.gateway_url,
            code=args.code,
            executor_key=executor_key,
            executor_display_name=executor_display_name,
            executor_capabilities=claim_capabilities,
        )
    except ValueError as e:
        logger.error("Claim failed: %s", e)
        sys.exit(1)

    logger.info("Agent created: %s (id=%s)", result.display_name, result.agent_id)

    creds_path = save_credentials(result)
    logger.info("Credentials saved to %s", creds_path)

    if args.no_start:
        logger.info("Done. Use --no-start=false or run executor separately.")
        return

    logger.info("Starting executor bridge...")

    executor = ExecutorClient(
        base_url=args.gateway_url,
        agent_id=result.agent_id,
        api_key=result.api_key,
        executor_key=executor_key,
        display_name=executor_display_name,
        capabilities=claim_capabilities or ["code"],
    )

    @executor.on_task
    async def handle_task(task: GatewayTask) -> dict[str, Any]:
        logger.info("Received task: %s (id=%s)", task.title, task.task_id)
        return {"summary": f"Task received: {task.title}", "status": "acknowledged"}

    logger.info("Executor running. Press Ctrl+C to stop.")
    executor.run()


async def _cmd_info(args: argparse.Namespace) -> None:
    from .invite import get_invite_info

    try:
        info = await get_invite_info(args.gateway_url, args.code)
    except ValueError as e:
        logger.error("%s", e)
        sys.exit(1)

    print(json.dumps(info, indent=2))


def _cmd_status() -> None:
    from .invite import load_credentials

    creds = load_credentials()
    if creds is None:
        print("No saved credentials found.")
        print("Use 'python -m agentchat join <code>' to claim an invite.")
        sys.exit(1)

    print(f"Agent ID:     {creds.agent_id}")
    print(f"Display Name: {creds.display_name}")
    print(f"API Key:      {creds.api_key[:10]}...")
    print(f"Gateway URL:  {creds.gateway_url}")
    if creds.executor_id:
        print(f"Executor ID:  {creds.executor_id}")


def _default_executor_key() -> str:
    """Generate a default executor key from hostname."""
    import socket
    hostname = socket.gethostname().lower().replace(" ", "-")
    return f"cli-{hostname}"


if __name__ == "__main__":
    main()
