"""
Interactive and non-interactive CLI for the agent.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import litellm
from lmnr import Laminar, LaminarLiteLLMCallback
from prompt_toolkit import PromptSession

from agent.config import load_config
from agent.core.agent_loop import submission_loop
from agent.core.session import OpType
from agent.core.tools import ToolRouter
from agent.utils.reliability_checks import check_training_script_save_pattern
from agent.utils.terminal_display import (
    format_error,
    format_header,
    format_plan_display,
    format_separator,
    format_success,
    format_tool_call,
    format_tool_output,
    format_turn_complete,
)

litellm.drop_params = True


def _safe_get_args(arguments: dict) -> dict:
    """Safely extract args dict from arguments, handling cases where LLM passes string."""
    args = arguments.get("args", {})
    # Sometimes LLM passes args as string instead of dict
    if isinstance(args, str):
        return {}
    return args if isinstance(args, dict) else {}


lmnr_api_key = os.environ.get("LMNR_API_KEY")
if lmnr_api_key:
    try:
        Laminar.initialize(project_api_key=lmnr_api_key)
        litellm.callbacks = [LaminarLiteLLMCallback()]
        print("Laminar initialized")
    except Exception as e:
        print(f"Failed to initialize Laminar: {e}")


@dataclass
class Operation:
    """Operation to be executed by the agent"""

    op_type: OpType
    data: Optional[dict[str, Any]] = None


@dataclass
class Submission:
    """Submission to the agent loop"""

    id: str
    operation: Operation


@dataclass
class NonInteractiveRunState:
    """Mutable state for a single non-interactive run."""

    seq: int = 0
    tool_call_count: int = 0
    tool_error_count: int = 0
    final_assistant_message: Optional[str] = None
    success: bool = False
    error_seen: bool = False
    done_event: asyncio.Event = field(default_factory=asyncio.Event)


def _write_jsonl_line(
    path: Path,
    obj: dict[str, Any],
) -> None:
    """Append one JSON line to path and flush."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


async def _non_interactive_event_listener(
    event_queue: asyncio.Queue,
    submission_queue: asyncio.Queue,
    session_id: str,
    run_state: NonInteractiveRunState,
    output_file: Optional[Path],
    verbose: bool,
    ready_event: asyncio.Event,
) -> None:
    """
    Consume events in non-interactive mode: write JSONL, auto-approve approvals,
    track counts and final message, signal when turn completes or errors.
    """
    while True:
        try:
            event = await event_queue.get()

            if event.event_type == "ready":
                run_state.done_event.clear()
                ready_event.set()
            elif event.event_type == "assistant_message" and event.data:
                run_state.final_assistant_message = event.data.get("content") or ""
            elif event.event_type == "tool_call":
                run_state.tool_call_count += 1
            elif event.event_type == "tool_output" and event.data:
                if not event.data.get("success", False):
                    run_state.tool_error_count += 1
            elif event.event_type == "turn_complete":
                if not run_state.error_seen:
                    run_state.success = True
                run_state.done_event.set()
            elif event.event_type == "error":
                run_state.error_seen = True
                run_state.success = False
                run_state.done_event.set()
            elif event.event_type == "approval_required" and event.data:
                tools_data = event.data.get("tools", []) or []
                approvals = [
                    {"tool_call_id": t.get("tool_call_id", ""), "approved": True, "feedback": None}
                    for t in tools_data
                ]
                submission = Submission(
                    id="approval_noninteractive",
                    operation=Operation(op_type=OpType.EXEC_APPROVAL, data={"approvals": approvals}),
                )
                await submission_queue.put(submission)
            elif event.event_type == "shutdown":
                break

            if output_file is not None:
                run_state.seq += 1
                line_obj = {
                    "ts": time.time(),
                    "seq": run_state.seq,
                    "event_type": event.event_type,
                    "data": event.data,
                    "session_id": session_id,
                    "mode": "non_interactive",
                }
                _write_jsonl_line(output_file, line_obj)

            if verbose and event.event_type in (
                "tool_call",
                "tool_output",
                "assistant_message",
                "turn_complete",
                "error",
            ):
                print(f"[{event.event_type}]", file=sys.stderr)

        except asyncio.CancelledError:
            break
        except Exception as e:
            if verbose:
                print(f"Event listener error: {e}", file=sys.stderr)
            run_state.success = False
            run_state.done_event.set()


async def run_non_interactive(args: argparse.Namespace) -> int:
    """
    One-shot run: submit prompt, auto-approve tool calls, stream events to file, exit.
    Returns exit code (0 on success).
    """
    config_path = args.config
    if config_path is None:
        config_path = str(Path(__file__).parent.parent / "configs" / "main_agent_config.json")
    config = load_config(config_path)
    if args.model is not None:
        config.model_name = args.model
    config.yolo_mode = True

    output_file = Path(args.output_file) if args.output_file else None
    if output_file is None and args.output_format == "stream-json":
        output_file = Path(os.getcwd()) / "hf_agent_events.jsonl"

    submission_queue: asyncio.Queue = asyncio.Queue()
    event_queue: asyncio.Queue = asyncio.Queue()
    ready_event = asyncio.Event()
    run_state = NonInteractiveRunState()

    tool_router = ToolRouter(config.mcpServers)
    agent_task = asyncio.create_task(
        submission_loop(
            submission_queue,
            event_queue,
            config=config,
            tool_router=tool_router,
        )
    )

    session_id = "noninteractive"
    listener_task = asyncio.create_task(
        _non_interactive_event_listener(
            event_queue,
            submission_queue,
            session_id,
            run_state,
            output_file,
            args.verbose,
            ready_event,
        )
    )

    await ready_event.wait()
    start = time.perf_counter()

    await submission_queue.put(
        Submission(
            id="sub_1",
            operation=Operation(op_type=OpType.USER_INPUT, data={"text": args.prompt}),
        )
    )

    await run_state.done_event.wait()
    duration_ms = int((time.perf_counter() - start) * 1000)
    exit_code = 0 if run_state.success else 1

    if output_file is not None:
        summary = {
            "ts": time.time(),
            "seq": run_state.seq + 1,
            "event_type": "run_complete",
            "mode": "non_interactive",
            "session_id": session_id,
            "data": {
                "success": run_state.success,
                "duration_ms": duration_ms,
                "exit_code": exit_code,
                "final_assistant_message": run_state.final_assistant_message,
                "tool_call_count": run_state.tool_call_count,
                "tool_error_count": run_state.tool_error_count,
            },
        }
        _write_jsonl_line(output_file, summary)

    await submission_queue.put(
        Submission(id="sub_shutdown", operation=Operation(op_type=OpType.SHUTDOWN))
    )
    try:
        await asyncio.wait_for(agent_task, timeout=10.0)
    except asyncio.TimeoutError:
        agent_task.cancel()
        try:
            await agent_task
        except asyncio.CancelledError:
            pass
    listener_task.cancel()
    try:
        await listener_task
    except asyncio.CancelledError:
        pass

    return exit_code


async def event_listener(
    event_queue: asyncio.Queue,
    submission_queue: asyncio.Queue,
    turn_complete_event: asyncio.Event,
    ready_event: asyncio.Event,
    prompt_session: PromptSession,
    config=None,
) -> None:
    """Background task that listens for events and displays them"""
    submission_id = [1000]  # Use list to make it mutable in closure
    last_tool_name = [None]  # Track last tool called

    while True:
        try:
            event = await event_queue.get()

            # Display event
            if event.event_type == "ready":
                print(format_success("\U0001f917 Agent ready"))
                ready_event.set()
            elif event.event_type == "assistant_message":
                content = event.data.get("content", "") if event.data else ""
                if content:
                    print(f"\nAssistant: {content}")
            elif event.event_type == "tool_call":
                tool_name = event.data.get("tool", "") if event.data else ""
                arguments = event.data.get("arguments", {}) if event.data else {}
                if tool_name:
                    last_tool_name[0] = tool_name  # Store for tool_output event
                    args_str = json.dumps(arguments)[:100] + "..."
                    print(format_tool_call(tool_name, args_str))
            elif event.event_type == "tool_output":
                output = event.data.get("output", "") if event.data else ""
                success = event.data.get("success", False) if event.data else False
                if output:
                    # Don't truncate plan_tool output, truncate everything else
                    should_truncate = last_tool_name[0] != "plan_tool"
                    print(format_tool_output(output, success, truncate=should_truncate))
            elif event.event_type == "turn_complete":
                print(format_turn_complete())
                # Display plan after turn complete
                plan_display = format_plan_display()
                if plan_display:
                    print(plan_display)
                turn_complete_event.set()
            elif event.event_type == "error":
                error = (
                    event.data.get("error", "Unknown error")
                    if event.data
                    else "Unknown error"
                )
                print(format_error(error))
                turn_complete_event.set()
            elif event.event_type == "shutdown":
                break
            elif event.event_type == "processing":
                pass  # print("Processing...", flush=True)
            elif event.event_type == "compacted":
                old_tokens = event.data.get("old_tokens", 0) if event.data else 0
                new_tokens = event.data.get("new_tokens", 0) if event.data else 0
                print(f"Compacted context: {old_tokens} → {new_tokens} tokens")
            elif event.event_type == "approval_required":
                # Handle batch approval format
                tools_data = event.data.get("tools", []) if event.data else []
                count = event.data.get("count", 0) if event.data else 0

                # If yolo mode is active, auto-approve everything
                if config and config.yolo_mode:
                    approvals = [
                        {
                            "tool_call_id": t.get("tool_call_id", ""),
                            "approved": True,
                            "feedback": None,
                        }
                        for t in tools_data
                    ]
                    print(f"\n⚡ YOLO MODE: Auto-approving {count} item(s)")
                    submission_id[0] += 1
                    approval_submission = Submission(
                        id=f"approval_{submission_id[0]}",
                        operation=Operation(
                            op_type=OpType.EXEC_APPROVAL,
                            data={"approvals": approvals},
                        ),
                    )
                    await submission_queue.put(approval_submission)
                    continue

                print("\n" + format_separator())
                print(
                    format_header(
                        f"APPROVAL REQUIRED ({count} item{'s' if count != 1 else ''})"
                    )
                )
                print(format_separator())

                approvals = []

                # Ask for approval for each tool
                for i, tool_info in enumerate(tools_data, 1):
                    tool_name = tool_info.get("tool", "")
                    arguments = tool_info.get("arguments", {})
                    tool_call_id = tool_info.get("tool_call_id", "")

                    # Handle case where arguments might be a JSON string
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            print(f"Warning: Failed to parse arguments for {tool_name}")
                            arguments = {}

                    operation = arguments.get("operation", "")

                    print(f"\n[Item {i}/{count}]")
                    print(f"Tool: {tool_name}")
                    print(f"Operation: {operation}")

                    # Handle different tool types
                    if tool_name == "hf_jobs":
                        # Check if this is Python mode (script) or Docker mode (command)
                        script = arguments.get("script")
                        command = arguments.get("command")

                        if script:
                            # Python mode
                            dependencies = arguments.get("dependencies", [])
                            python_version = arguments.get("python")
                            script_args = arguments.get("script_args", [])

                            # Show full script
                            print(f"Script:\n{script}")
                            if dependencies:
                                print(f"Dependencies: {', '.join(dependencies)}")
                            if python_version:
                                print(f"Python version: {python_version}")
                            if script_args:
                                print(f"Script args: {' '.join(script_args)}")

                            # Run reliability checks on the full script (not truncated)
                            check_message = check_training_script_save_pattern(script)
                            if check_message:
                                print(check_message)
                        elif command:
                            # Docker mode
                            image = arguments.get("image", "python:3.12")
                            command_str = (
                                " ".join(command)
                                if isinstance(command, list)
                                else str(command)
                            )
                            print(f"Docker image: {image}")
                            print(f"Command: {command_str}")

                        # Common parameters for jobs
                        hardware_flavor = arguments.get("hardware_flavor", "cpu-basic")
                        timeout = arguments.get("timeout", "30m")
                        env = arguments.get("env", {})
                        schedule = arguments.get("schedule")

                        print(f"Hardware: {hardware_flavor}")
                        print(f"Timeout: {timeout}")

                        if env:
                            env_keys = ", ".join(env.keys())
                            print(f"Environment variables: {env_keys}")

                        if schedule:
                            print(f"Schedule: {schedule}")

                    elif tool_name == "hf_private_repos":
                        # Handle private repo operations
                        args = _safe_get_args(arguments)

                        if operation in ["create_repo", "upload_file"]:
                            repo_id = args.get("repo_id", "")
                            repo_type = args.get("repo_type", "dataset")

                            # Build repo URL
                            type_path = "" if repo_type == "model" else f"{repo_type}s"
                            repo_url = (
                                f"https://huggingface.co/{type_path}/{repo_id}".replace(
                                    "//", "/"
                                )
                            )

                            print(f"Repository: {repo_id}")
                            print(f"Type: {repo_type}")
                            print("Private: Yes")
                            print(f"URL: {repo_url}")

                            # Show file preview for upload_file operation
                            if operation == "upload_file":
                                path_in_repo = args.get("path_in_repo", "")
                                file_content = args.get("file_content", "")
                                print(f"File: {path_in_repo}")

                                if isinstance(file_content, str):
                                    # Calculate metrics
                                    all_lines = file_content.split("\n")
                                    line_count = len(all_lines)
                                    size_bytes = len(file_content.encode("utf-8"))
                                    size_kb = size_bytes / 1024
                                    size_mb = size_kb / 1024

                                    print(f"Line count: {line_count}")
                                    if size_kb < 1024:
                                        print(f"Size: {size_kb:.2f} KB")
                                    else:
                                        print(f"Size: {size_mb:.2f} MB")

                                    # Show preview
                                    preview_lines = all_lines[:5]
                                    preview = "\n".join(preview_lines)
                                    print(
                                        f"Content preview (first 5 lines):\n{preview}"
                                    )
                                    if len(all_lines) > 5:
                                        print("...")

                    elif tool_name == "hf_repo_files":
                        # Handle repo files operations (upload, delete)
                        repo_id = arguments.get("repo_id", "")
                        repo_type = arguments.get("repo_type", "model")
                        revision = arguments.get("revision", "main")

                        # Build repo URL
                        if repo_type == "model":
                            repo_url = f"https://huggingface.co/{repo_id}"
                        else:
                            repo_url = f"https://huggingface.co/{repo_type}s/{repo_id}"

                        print(f"Repository: {repo_id}")
                        print(f"Type: {repo_type}")
                        print(f"Branch: {revision}")
                        print(f"URL: {repo_url}")

                        if operation == "upload":
                            path = arguments.get("path", "")
                            content = arguments.get("content", "")
                            create_pr = arguments.get("create_pr", False)

                            print(f"File: {path}")
                            if create_pr:
                                print("Mode: Create PR")

                            if isinstance(content, str):
                                all_lines = content.split("\n")
                                line_count = len(all_lines)
                                size_bytes = len(content.encode("utf-8"))
                                size_kb = size_bytes / 1024

                                print(f"Lines: {line_count}")
                                if size_kb < 1024:
                                    print(f"Size: {size_kb:.2f} KB")
                                else:
                                    print(f"Size: {size_kb / 1024:.2f} MB")

                                # Show full content
                                print(f"Content:\n{content}")

                        elif operation == "delete":
                            patterns = arguments.get("patterns", [])
                            if isinstance(patterns, str):
                                patterns = [patterns]
                            print(f"Patterns to delete: {', '.join(patterns)}")

                    elif tool_name == "hf_repo_git":
                        # Handle git operations (branches, tags, PRs, repo management)
                        repo_id = arguments.get("repo_id", "")
                        repo_type = arguments.get("repo_type", "model")

                        # Build repo URL
                        if repo_type == "model":
                            repo_url = f"https://huggingface.co/{repo_id}"
                        else:
                            repo_url = f"https://huggingface.co/{repo_type}s/{repo_id}"

                        print(f"Repository: {repo_id}")
                        print(f"Type: {repo_type}")
                        print(f"URL: {repo_url}")

                        if operation == "delete_branch":
                            branch = arguments.get("branch", "")
                            print(f"Branch to delete: {branch}")

                        elif operation == "delete_tag":
                            tag = arguments.get("tag", "")
                            print(f"Tag to delete: {tag}")

                        elif operation == "merge_pr":
                            pr_num = arguments.get("pr_num", "")
                            print(f"PR to merge: #{pr_num}")

                        elif operation == "create_repo":
                            private = arguments.get("private", False)
                            space_sdk = arguments.get("space_sdk")
                            print(f"Private: {private}")
                            if space_sdk:
                                print(f"Space SDK: {space_sdk}")

                        elif operation == "update_repo":
                            private = arguments.get("private")
                            gated = arguments.get("gated")
                            if private is not None:
                                print(f"Private: {private}")
                            if gated is not None:
                                print(f"Gated: {gated}")

                    # Get user decision for this item
                    response = await prompt_session.prompt_async(
                        f"Approve item {i}? (y=yes, yolo=approve all, n=no, or provide feedback): "
                    )

                    response = response.strip().lower()

                    # Handle yolo mode activation
                    if response == "yolo":
                        config.yolo_mode = True
                        print(
                            "⚡ YOLO MODE ACTIVATED - Auto-approving all future tool calls"
                        )
                        # Auto-approve this item and all remaining
                        approvals.append(
                            {
                                "tool_call_id": tool_call_id,
                                "approved": True,
                                "feedback": None,
                            }
                        )
                        for remaining in tools_data[i:]:
                            approvals.append(
                                {
                                    "tool_call_id": remaining.get("tool_call_id", ""),
                                    "approved": True,
                                    "feedback": None,
                                }
                            )
                        break

                    approved = response in ["y", "yes"]
                    feedback = None if approved or response in ["n", "no"] else response

                    approvals.append(
                        {
                            "tool_call_id": tool_call_id,
                            "approved": approved,
                            "feedback": feedback,
                        }
                    )

                # Submit batch approval
                submission_id[0] += 1
                approval_submission = Submission(
                    id=f"approval_{submission_id[0]}",
                    operation=Operation(
                        op_type=OpType.EXEC_APPROVAL,
                        data={"approvals": approvals},
                    ),
                )
                await submission_queue.put(approval_submission)
                print(format_separator() + "\n")
            # Silently ignore other events

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Event listener error: {e}")


async def get_user_input(prompt_session: PromptSession) -> str:
    """Get user input asynchronously"""
    from prompt_toolkit.formatted_text import HTML

    return await prompt_session.prompt_async(HTML("\n<b><cyan>></cyan></b> "))


async def main():
    """Interactive chat with the agent"""
    from agent.utils.terminal_display import Colors

    # Clear screen
    os.system("clear" if os.name != "nt" else "cls")

    banner = r"""
  _   _                   _               _____                   _                    _   
 | | | |_   _  __ _  __ _(_)_ __   __ _  |  ___|_ _  ___ ___     / \   __ _  ___ _ __ | |_ 
 | |_| | | | |/ _` |/ _` | | '_ \ / _` | | |_ / _` |/ __/ _ \   / _ \ / _` |/ _ \ '_ \| __|
 |  _  | |_| | (_| | (_| | | | | | (_| | |  _| (_| | (_|  __/  / ___ \ (_| |  __/ | | | |_ 
 |_| |_|\__,_|\__, |\__, |_|_| |_|\__, | |_|  \__,_|\___\___| /_/   \_\__, |\___|_| |_|\__|
              |___/ |___/         |___/                               |___/
    """

    print(format_separator())
    print(f"{Colors.YELLOW} {banner}{Colors.RESET}")
    print("Type your messages below. Type 'exit', 'quit', or '/quit' to end.\n")
    print(format_separator())
    # Wait for agent to initialize
    print("Initializing agent...")

    # Create queues for communication
    submission_queue = asyncio.Queue()
    event_queue = asyncio.Queue()

    # Events to signal agent state
    turn_complete_event = asyncio.Event()
    turn_complete_event.set()
    ready_event = asyncio.Event()

    # Start agent loop in background
    config_path = Path(__file__).parent.parent / "configs" / "main_agent_config.json"
    config = load_config(config_path)

    # Create tool router
    print(f"Loading MCP servers: {', '.join(config.mcpServers.keys())}")
    tool_router = ToolRouter(config.mcpServers)

    # Create prompt session for input
    prompt_session = PromptSession()

    agent_task = asyncio.create_task(
        submission_loop(
            submission_queue,
            event_queue,
            config=config,
            tool_router=tool_router,
        )
    )

    # Start event listener in background
    listener_task = asyncio.create_task(
        event_listener(
            event_queue,
            submission_queue,
            turn_complete_event,
            ready_event,
            prompt_session,
            config,
        )
    )

    await ready_event.wait()

    submission_id = 0

    try:
        while True:
            # Wait for previous turn to complete
            await turn_complete_event.wait()
            turn_complete_event.clear()

            # Get user input
            try:
                user_input = await get_user_input(prompt_session)
            except EOFError:
                break

            # Check for exit commands
            if user_input.strip().lower() in ["exit", "quit", "/quit", "/exit"]:
                break

            # Skip empty input
            if not user_input.strip():
                turn_complete_event.set()
                continue

            # Submit to agent
            submission_id += 1
            submission = Submission(
                id=f"sub_{submission_id}",
                operation=Operation(
                    op_type=OpType.USER_INPUT, data={"text": user_input}
                ),
            )
            # print(f"Main submitting: {submission.operation.op_type}")
            await submission_queue.put(submission)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    # Shutdown
    print("\n🛑 Shutting down agent...")
    shutdown_submission = Submission(
        id="sub_shutdown", operation=Operation(op_type=OpType.SHUTDOWN)
    )
    await submission_queue.put(shutdown_submission)

    await asyncio.wait_for(agent_task, timeout=5.0)
    listener_task.cancel()

    print("✨ Goodbye!\n")


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments. Defaults to interactive mode when --prompt is not set."""
    parser = argparse.ArgumentParser(
        description="HF Agent CLI: interactive chat or one-shot prompt execution.",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default=None,
        help="Run in non-interactive mode with this prompt and exit after one turn.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="In non-interactive mode, stream JSON events to this file (one JSON object per line).",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["stream-json"],
        default=None,
        help="Output format for non-interactive mode; only 'stream-json' is supported.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name from config (e.g. litellm model string or path to config).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Emit extra progress to stderr in non-interactive mode.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to agent config JSON (default: configs/main_agent_config.json).",
    )
    return parser.parse_args(args)


def cli() -> None:
    """Sync entry point for the installed `hf-agent` console script."""
    parsed = parse_args()
    if parsed.prompt is not None:
        exit_code = asyncio.run(run_non_interactive(parsed))
        sys.exit(exit_code)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n✨ Goodbye!")


if __name__ == "__main__":
    cli()
