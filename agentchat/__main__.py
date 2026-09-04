"""AgentChat CLI — python -m agentchat <command>

Commands:
    join <code>      Claim an invite, save credentials, start executor
    connect <code>   Claim an invite as an EXTERNAL agent (your own Claude
                     Code / Codex session) and print the CLI configuration
    info <code>      Show public invite info
    status           Show saved credentials
"""

from __future__ import annotations

import argparse
from pathlib import Path
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
    join_parser.add_argument("--no-start", action="store_true", help="Don't start the executor gateway loop")

    # connect (external agent, #148)
    connect_parser = subparsers.add_parser(
        "connect",
        help="Claim an invite as an external agent and print the Claude Code / Codex config",
    )
    connect_parser.add_argument("code", help="Invite code (e.g. inv_Abc123)")
    connect_parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL, help="Backend URL")
    connect_parser.add_argument(
        "--tool",
        choices=["claude-code", "codex"],
        default="claude-code",
        help="Which CLI you drive (default: claude-code)",
    )
    connect_parser.add_argument(
        "--install",
        action="store_true",
        help="Claude Code only: run `claude mcp add` and merge the hooks into ~/.claude/settings.json",
    )
    connect_parser.add_argument(
        "--project",
        nargs="?",
        const=".",
        default=None,
        metavar="DIR",
        help=(
            "Bind this agent to one repo (default: the current folder) instead of making it "
            "the machine's default: credentials go to <DIR>/.claude/agntchat and the MCP "
            "server is registered in Claude Code's local scope for that folder"
        ),
    )

    # sessions (desktop picker, #148): Claude Code sessions on this machine
    sessions_parser = subparsers.add_parser(
        "sessions", help="List Claude Code sessions on this machine as JSON (external agents)"
    )
    sessions_parser.add_argument("--limit", type=int, default=30)

    # bind (desktop picker, #148): point one Claude Code session at one agent
    bind_parser = subparsers.add_parser(
        "bind", help="Bind a Claude Code session to an external agent (writes local files only)"
    )
    bind_parser.add_argument("--session", required=True, help="Claude Code session id")
    bind_parser.add_argument("--agent-id", required=True)
    bind_parser.add_argument(
        "--api-key",
        default=None,
        help="The agent's key; omit to reuse the credentials this machine already holds for it",
    )
    bind_parser.add_argument("--display-name", default="")
    bind_parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
    bind_parser.add_argument("--cwd", default=None, help="The session's working directory")
    bind_parser.add_argument(
        "--title", default=None, help="Name the session (applied by the hook on its first prompt)"
    )
    bind_parser.add_argument(
        "--conversation", default=None, help="The session conversation the server linked (#148)"
    )
    bind_parser.add_argument(
        "--install",
        action="store_true",
        help="Also make sure the user-scope hooks and MCP server are installed",
    )

    # identities (desktop, #148): external agents this machine holds credentials for
    subparsers.add_parser(
        "identities", help="List external agents with credentials saved on this machine (JSON)"
    )

    # rekey (desktop, #148): the agent's API key was regenerated — refresh
    # the local binding credentials so bound sessions keep working
    rekey_parser = subparsers.add_parser(
        "rekey", help="Refresh the saved credentials for a session-bound external agent"
    )
    rekey_parser.add_argument("--agent-id", required=True)
    rekey_parser.add_argument("--api-key", required=True)
    rekey_parser.add_argument("--display-name", default="")
    rekey_parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)

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
    elif args.command == "connect":
        asyncio.run(_cmd_connect(args))
    elif args.command == "sessions":
        _cmd_sessions(args)
    elif args.command == "bind":
        _cmd_bind(args)
    elif args.command == "rekey":
        _cmd_rekey(args)
    elif args.command == "identities":
        _cmd_identities()
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


async def _cmd_connect(args: argparse.Namespace) -> None:
    """External agent (#148): claim the invite with `runtime: external` (no
    executor is registered — nothing will ever be pushed to this agent), save
    credentials, and print or install the CLI configuration."""
    from . import external
    from .invite import claim_invite, save_credentials

    tool = "codex" if args.tool == "codex" else "claude_code"
    logger.info("Claiming invite %s as an external %s agent...", args.code, args.tool)

    try:
        result = await claim_invite(
            gateway_url=args.gateway_url,
            code=args.code,
            runtime="external",
            external_tool=tool,
        )
    except ValueError as e:
        logger.error("Claim failed: %s", e)
        sys.exit(1)

    logger.info("Connected as: %s (id=%s)", result.display_name, result.agent_id)
    project = Path(args.project).resolve() if args.project else None
    if project:
        home = external.project_home(project)
        external.write_project_binding_ignore(home)
        creds_path = save_credentials(result, home)
        logger.info("Project binding: %s → %s", project, creds_path)
    else:
        creds_path = save_credentials(result)
        logger.info("Credentials saved to %s (this machine's default agent)", creds_path)

    print()
    if args.install and tool == "claude_code":
        for step in external.install_claude(project=project):
            print(f"  • {step}")
        print()
        print("Start a new session — for live chat from agntchat, start it as a channel:")
        print()
        print(f"  {external.claude_channels_command()}")
        print()
        print("(A plain `claude` works too; messages then arrive before each prompt and")
        print("at the end of each turn.) It appears in agntchat as an External agent.")
    else:
        print(external.render_instructions(tool, project=project))
    print()


def _cmd_sessions(args: argparse.Namespace) -> None:
    """JSON for the desktop session picker."""
    from . import external

    print(json.dumps(external.list_claude_sessions(limit=args.limit)))


def _cmd_bind(args: argparse.Namespace) -> None:
    """Bind one Claude Code session to one agent. The desktop app already
    holds the agent's fresh API key (it asked the backend), so this only
    writes local files: the agent's credentials folder, the session map,
    and — with --install — the user-scope hooks/MCP registration."""
    from . import external
    from .invite import ClaimResult, save_credentials

    home = external.agent_home(args.agent_id)
    if args.api_key:
        save_credentials(
            ClaimResult(
                agent_id=args.agent_id,
                api_key=args.api_key,
                gateway_url=args.gateway_url,
                display_name=args.display_name or args.agent_id,
            ),
            home,
        )
    elif not external._read_credentials_at(home):
        # Reuse the machine default when it is this agent; otherwise there is
        # nothing to bind with — regenerating a key here would invalidate the
        # copies every other session holds (that rotation loop silenced the
        # identity on 2026-09-04).
        default = external._read_credentials_at(external._HOME)
        if default and default.get("agent_id") == args.agent_id:
            save_credentials(
                ClaimResult(
                    agent_id=args.agent_id,
                    api_key=default["api_key"],
                    gateway_url=default.get("gateway_url") or args.gateway_url,
                    display_name=default.get("display_name") or args.display_name or args.agent_id,
                ),
                home,
            )
        else:
            print(json.dumps({"error": "no credentials for this agent on this machine; pass --api-key"}))
            sys.exit(2)
    entry = external.bind_session(
        args.session, args.agent_id, cwd=args.cwd, title=args.title, conversation_id=args.conversation
    )
    steps: list[str] = []
    if args.install:
        steps = external.install_claude()
    print(json.dumps({"session": args.session, "binding": entry, "home": str(home), "steps": steps}))


def _cmd_identities() -> None:
    """External agents this machine can act as: the machine default plus
    every per-session agent home with credentials."""
    from . import external

    seen: dict[str, dict[str, Any]] = {}
    default = external._read_credentials_at(external._HOME)
    if default:
        seen[default["agent_id"]] = {"agentId": default["agent_id"], "displayName": default.get("display_name"), "default": True}
    try:
        for home in external.AGENTS_DIR.iterdir():
            creds = external._read_credentials_at(home)
            if creds and creds["agent_id"] not in seen:
                seen[creds["agent_id"]] = {"agentId": creds["agent_id"], "displayName": creds.get("display_name"), "default": False}
    except OSError:
        pass
    print(json.dumps(list(seen.values())))


def _cmd_rekey(args: argparse.Namespace) -> None:
    """Regenerating an external agent's key invalidates the credentials the
    picker saved; rewrite them (agent home + any project binding for this
    agent + the machine default if it is this agent) so nothing goes stale.
    Also clears the hook's cached token / backoff for that agent."""
    from . import external
    from .invite import ClaimResult, save_credentials

    result = ClaimResult(
        agent_id=args.agent_id,
        api_key=args.api_key,
        gateway_url=args.gateway_url,
        display_name=args.display_name or args.agent_id,
    )
    homes = [external.agent_home(args.agent_id)]
    default = external._read_credentials_at(external._HOME)
    if default and default.get("agent_id") == args.agent_id:
        homes.append(external._HOME)
    for entry in external.load_session_bindings().values():
        cwd = entry.get("cwd") if isinstance(entry, dict) else None
        home = external.find_project_home(cwd) if cwd else None
        creds = external._read_credentials_at(home) if home else None
        if creds and creds.get("agent_id") == args.agent_id and home not in homes:
            homes.append(home)
    written = []
    for home in homes:
        written.append(str(save_credentials(result, home)))
        try:
            (home / "hook-token.json").unlink()
        except FileNotFoundError:
            pass
    print(json.dumps({"agent": args.agent_id, "written": written}))


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
