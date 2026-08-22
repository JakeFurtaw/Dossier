"""TraceBus (pure run state) + RichRenderer (live console).

Agents talk only to the bus. A renderer — or later a web UI / test harness —
subscribes to the same events without the other side knowing.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from workflow.config import HOST, MODEL, MODEL_EVALUATOR, MODEL_PLANNER, MODEL_RESEARCHER, NUM_PREDICT, OBS_TRUNCATE, REPORT_DIR, TEMPERATURE
from workflow.runtime.ledger import LedgerEntry, SharedContext
from workflow.runtime.metrics import Counters, install_counters, snapshot, uninstall_counters
from workflow.runtime.report import TraceEvent, write_report

logger = logging.getLogger(__name__)

_current: ContextVar[TraceBus | None] = ContextVar("trace_bus", default=None)
_agent_stack: ContextVar[tuple[str, ...]] = ContextVar("agent_stack", default=())
_fallback: TraceBus | None = None

def _role_color(role: str) -> str:
    """Console color for an agent role from the active recipe (white fallback)."""
    try:
        from workflow.recipes import active_recipe

        color = active_recipe().role_colors.get(role)
        if color:
            return color
    except Exception:
        pass
    return "white"


def _preview(text: str, limit: int, verbose: bool) -> str:
    """Truncate long text for panels; tells the reader where the rest lives."""
    text = text or ""
    if verbose or len(text) <= limit:
        return text
    hidden = len(text) - limit
    return text[:limit].rstrip() + f"\n… [{hidden} more chars — full text in the run report]"


def _one_line(text: str, limit: int = 88) -> str:
    """Collapse whitespace to a single bounded line (compact action rows)."""
    line = " ".join((text or "").split())
    if len(line) <= limit:
        return line
    return line[: limit - 1] + "…"


def _observation_summary(text: str) -> str:
    """One-line verdict for a tool output (verdict, hits, blocked, N reports, size)."""
    match = re.search(r"\*\*Verdict:\*\*\s*(PASS|WEAK|FAIL)", text or "", flags=re.I)
    if match:
        return f"evaluator {match.group(1).upper()}"
    match = re.search(r"Search results for .+ \((\d+) hits\)", text or "")
    if match:
        return f"{match.group(1)} search hits"
    if (text or "").startswith("Blocked:"):
        return "page blocked"
    if "### Parallel" in (text or ""):
        return f"{(text or '').count('### Parallel') or 1} parallel report(s)"
    chars = len(text or "")
    return f"ok · {chars} chars"


def _tool_detail(name: str, args: dict[str, Any]) -> str:
    """'tool · short-arg' label for compact action rows and the live tree."""
    for key in ("query", "url", "task", "expression"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            short = value.strip().replace("\n", " ")
            if len(short) > 72:
                short = short[:69] + "…"
            return f"{name} · {short}"
    return name


@dataclass
class AgentNode:
    agent_id: str
    role: str
    max_iterations: int = 0
    step: int = 0
    status: str = "idle"
    activity: str = ""
    children: list[AgentNode] = field(default_factory=list)
    parent_id: str = ""
    depth: int = 0

    @property
    def busy(self) -> bool:
        """True while the agent is mid-thought / acting / waiting on a child."""
        return self.status in {"thinking", "acting", "waiting"}


class TraceListener:
    """Optional subscriber. Override the hooks you care about."""

    def on_start(self, bus: TraceBus) -> None:
        """Run starting (before any events)."""
        return None

    def on_event(self, bus: TraceBus, event: TraceEvent) -> None:
        """One thought/action/observation/note/spawn/finish event."""
        return None

    def on_tree(self, bus: TraceBus) -> None:
        """Agent tree changed; re-render status if wanted."""
        return None

    def on_complete(self, bus: TraceBus, citation_summary: str = "") -> None:
        """Run finished; optional one-line citation verdict available."""
        return None

    def on_stop(self, bus: TraceBus) -> None:
        """Bus torn down (always last)."""
        return None


class TraceBus:
    """Source of truth for one run: events, agent tree, URL cache. No console."""

    def __init__(
        self,
        *,
        goal: str = "",
        verbose: bool = False,
        save: bool = True,
        report_dir: str | Path = REPORT_DIR,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Configure one run: goal, save behavior, default report metadata."""
        self.goal = goal
        self.verbose = verbose
        self.save = save
        self.report_dir = Path(report_dir)
        self.config = config or {
            "model": MODEL,
            "model_planner": MODEL_PLANNER,
            "model_researcher": MODEL_RESEARCHER,
            "model_evaluator": MODEL_EVALUATOR,
            "host": HOST,
            "temperature": TEMPERATURE,
            "num_predict": NUM_PREDICT,
        }
        self.events: list[TraceEvent] = []
        self.roots: list[AgentNode] = []
        self.nodes: dict[str, AgentNode] = {}
        self._role_counts: dict[str, int] = {}
        self._lock = threading.RLock()
        self._listeners: list[TraceListener] = []
        self.shared = SharedContext()
        self.started = datetime.now().astimezone()
        self.final = ""
        self.reason = ""
        self.citation_audit_md = ""
        self.report_path: Path | None = None
        self._completed = False

    def subscribe(self, listener: TraceListener) -> None:
        """Register a subscriber (RichRenderer, or tests / a future web UI)."""
        self._listeners.append(listener)

    def _notify(self, method: str, *args: Any) -> None:
        """Fan out one hook to all subscribers."""
        for listener in list(self._listeners):
            getattr(listener, method)(self, *args)

    def start(self) -> None:
        """Signal 'run started' (renderer shows header + live tree)."""
        self._notify("on_start")

    def stop(self) -> None:
        """Signal 'run ended' (renderer stops the live view)."""
        self._notify("on_stop")

    def complete(
        self,
        *,
        final: str = "",
        reason: str = "",
        citation_audit_md: str = "",
        citation_summary: str = "",
    ) -> None:
        """End the run: record final/reason, write the report once, notify listeners."""
        self.final = final or self.final
        self.reason = reason or self.reason
        self.citation_audit_md = citation_audit_md or self.citation_audit_md
        self._notify("on_tree")
        if self.save and not self._completed:
            self._write_report()
        self._completed = True
        self._notify("on_complete", citation_summary)

    def start_agent(
        self,
        role: str,
        max_iterations: int = 0,
        parent_id: str | None = None,
    ) -> str:
        """Create this thread's agent node (nested under the current one) and emit spawn."""
        with self._lock:
            n = self._role_counts.get(role, 0) + 1
            self._role_counts[role] = n
            agent_id = f"{role}-{n}"
            stack = list(_agent_stack.get())
            attach_to = parent_id or (stack[-1] if stack else None)
            self._attach_agent(agent_id, role, max_iterations=max_iterations, parent_id=attach_to)
            stack.append(agent_id)
            _agent_stack.set(tuple(stack))
        self._emit(TraceEvent(kind="spawn", role=role, agent_id=agent_id, step=0, text=role))
        return agent_id

    def end_agent(self, agent_id: str, reason: str) -> None:
        """Mark the agent done, detach it from the thread stack, emit finish."""
        with self._lock:
            node = self.nodes.get(agent_id)
            self._mark_done(agent_id, reason)
            stack = list(_agent_stack.get())
            if stack and stack[-1] == agent_id:
                stack.pop()
                _agent_stack.set(tuple(stack))
        self._emit(
            TraceEvent(
                kind="finish",
                role=node.role if node else "",
                agent_id=agent_id,
                step=0,
                text=reason,
            )
        )

    def set_step(self, agent_id: str, step: int) -> None:
        """Advance an agent's step counter (live tree + report headings)."""
        with self._lock:
            node = self.nodes.get(agent_id)
            if node:
                node.step = step
        self._notify("on_tree")

    @contextmanager
    def thinking(self, agent_id: str) -> Iterator[None]:
        """Mark the agent as thinking (on Ollama) for the span's duration."""
        node = self.nodes.get(agent_id)
        if node:
            node.status = "thinking"
            node.activity = "waiting on Ollama"
            self._notify("on_tree")
        try:
            yield
        finally:
            if node and node.status == "thinking":
                node.status = "idle"
                node.activity = ""
                self._notify("on_tree")

    @contextmanager
    def acting(self, agent_id: str, tool: str, args: dict[str, Any]) -> Iterator[None]:
        """Mark the agent as acting (with tool detail) for the span's duration."""
        node = self.nodes.get(agent_id)
        if node:
            node.status = "acting"
            node.activity = _tool_detail(tool, args)
            self._notify("on_tree")
        try:
            yield
        finally:
            if node and node.status == "acting":
                node.status = "idle"
                self._notify("on_tree")

    def thought(self, agent_id: str, role: str, step: int, text: str) -> None:
        """Record a Thought event for the trace + report."""
        self._emit(TraceEvent(kind="thought", role=role, agent_id=agent_id, step=step, text=text))

    def action(self, agent_id: str, role: str, step: int, name: str, args: dict[str, Any]) -> None:
        """Record an Action (tool call) event."""
        self._emit(
            TraceEvent(kind="action", role=role, agent_id=agent_id, step=step, tool=name, args=args)
        )

    def observation(self, agent_id: str, role: str, step: int, text: str) -> None:
        """Record a tool's Observation event."""
        self._emit(TraceEvent(kind="observation", role=role, agent_id=agent_id, step=step, text=text))

    def note(self, agent_id: str, role: str, step: int, text: str) -> None:
        """Record a short diagnostic Note event (fallbacks, LLM errors, …)."""
        self._emit(TraceEvent(kind="note", role=role, agent_id=agent_id, step=step, text=text))

    def get_cached_url(self, url: str, kind: str = "page") -> str | None:
        """Cached page body for this run (web.py hit path)."""
        return self.shared.get_url(url, kind=kind)

    def put_cached_url(self, url: str, content: str, kind: str = "page") -> None:
        """Cache a fetched page body."""
        self.shared.put_url(url, content, kind=kind)

    def get_cached_search(self, query: str, max_results: int) -> str | None:
        """Cached web_search result."""
        return self.shared.get_search(query, max_results)

    def put_cached_search(self, query: str, max_results: int, content: str) -> None:
        """Cache a web_search result."""
        self.shared.put_search(query, max_results, content)

    def acquire_search(self, query: str, max_results: int) -> bool:
        """True when this thread may run the search (see SharedContext)."""
        return self.shared.acquire_search(query, max_results)

    def wait_search(self, query: str, max_results: int, timeout: float = 90.0) -> None:
        """Wait on another thread's in-flight run of the same search."""
        self.shared.wait_search(query, max_results, timeout=timeout)

    def release_search(self, query: str, max_results: int) -> None:
        """Finish a search and wake its waiters."""
        self.shared.release_search(query, max_results)

    def acquire_url_fetch(self, url: str, kind: str = "page") -> bool:
        """True when this thread may fetch the URL (see SharedContext)."""
        return self.shared.acquire_url(url, kind=kind)

    def wait_url_fetch(self, url: str, timeout: float = 60.0, kind: str = "page") -> None:
        """Wait on another thread's in-flight fetch of the same URL."""
        self.shared.wait_url(url, timeout=timeout, kind=kind)

    def release_url_fetch(self, url: str, kind: str = "page") -> None:
        """Finish a URL fetch and wake its waiters."""
        self.shared.release_url(url, kind=kind)

    def publish_entry(self, entry: LedgerEntry) -> None:
        """Add a ledger note (report/search/browse) for peer digests."""
        self.shared.publish(entry)

    def peer_digest(self, role: str, *, exclude_agent: str = "") -> str:
        """'Already gathered by other agents' block for a specialist's prompt."""
        return self.shared.digest(role, exclude_agent=exclude_agent)

    def ingest_event(self, event: TraceEvent) -> None:
        """Record a pre-built event and update the tree (used by --replay)."""
        with self._lock:
            if event.kind == "spawn":
                parent = self._replay_parent(event.role)
                if event.agent_id not in self.nodes:
                    role_key = event.role or "agent"
                    current = self._role_counts.get(role_key, 0)
                    suffix = event.agent_id.rsplit("-", 1)[-1]
                    if suffix.isdigit():
                        self._role_counts[role_key] = max(current, int(suffix))
                    else:
                        self._role_counts[role_key] = current + 1
                    self._attach_agent(event.agent_id, event.role, parent_id=parent)
            elif event.kind == "finish":
                self._mark_done(event.agent_id, event.text)
            elif event.agent_id:
                if event.agent_id not in self.nodes:
                    role = event.role or event.agent_id.rsplit("-", 1)[0]
                    self._attach_agent(
                        event.agent_id, role, parent_id=self._replay_parent(role)
                    )
                node = self.nodes.get(event.agent_id)
                if node:
                    node.step = max(node.step, event.step)
                    if event.kind == "thought":
                        node.status = "thinking"
                        node.activity = "replay"
                    elif event.kind == "action":
                        node.status = "acting"
                        node.activity = _tool_detail(event.tool, event.args)
                    elif event.kind == "observation":
                        node.status = "idle"
            self.events.append(event)
        self._notify("on_event", event)
        self._notify("on_tree")

    def depth(self, agent_id: str) -> int:
        """Tree depth of an agent (used for console indentation)."""
        node = self.nodes.get(agent_id)
        return node.depth if node else 0

    def _attach_agent(
        self,
        agent_id: str,
        role: str,
        *,
        max_iterations: int = 0,
        parent_id: str | None = None,
    ) -> AgentNode:
        """Create a node and attach it under its parent (or as a root)."""
        node = AgentNode(agent_id=agent_id, role=role, max_iterations=max_iterations)
        parent = self.nodes.get(parent_id) if parent_id else None
        if parent:
            parent.children.append(node)
            parent.status = "waiting"
            parent.activity = f"waiting on {role}"
        else:
            self.roots.append(node)
        node.parent_id = parent.agent_id if parent else ""
        node.depth = (parent.depth + 1) if parent else 0
        self.nodes[agent_id] = node
        return node

    def _mark_done(self, agent_id: str, reason: str) -> None:
        """Mark an agent done and resume its parent's status when all children finish."""
        node = self.nodes.get(agent_id)
        if node:
            node.status = "done"
            node.activity = reason
        parent = self.nodes.get(node.parent_id) if node and node.parent_id else None
        if parent and parent.status == "waiting":
            still_busy = any(
                child.busy for child in parent.children if child.agent_id != agent_id
            )
            if not still_busy:
                if node and node.role == "evaluator":
                    parent.status = "done"
                    parent.activity = f"evaluated {reason}"
                else:
                    parent.status = "acting"
                    parent.activity = parent.activity.replace(
                        "waiting on ", "resuming after "
                    )

    def _replay_parent(self, role: str) -> str | None:
        """Best-effort parent for a replayed event (planner roots itself)."""
        if role == "planner":
            return None
        if role == "evaluator":
            for node in reversed(list(self.nodes.values())):
                if node.role not in {"planner", "evaluator"}:
                    return node.agent_id
            return None
        for node in reversed(list(self.nodes.values())):
            if node.role == "planner":
                return node.agent_id
        if self.roots:
            return self.roots[-1].agent_id
        return None

    def _emit(self, event: TraceEvent) -> None:
        """Append an event and notify listeners (event + tree)."""
        with self._lock:
            self.events.append(event)
        self._notify("on_event", event)
        self._notify("on_tree")

    def _write_report(self) -> None:
        """Serialize this run to runs/<timestamp>.md (called once by complete)."""
        ended = datetime.now().astimezone()
        stem = self.started.strftime("%Y%m%d-%H%M%S")
        self.report_path = write_report(
            self.report_dir,
            stem,
            goal=self.goal,
            config=self.config,
            events=self.events,
            final=self.final,
            reason=self.reason,
            started=self.started,
            ended=ended,
            citation_audit_md=self.citation_audit_md,
            counters=snapshot(),
            shared_context_md=self.shared.to_markdown(),
        )


class RichRenderer(TraceListener):
    """Live Rich tree + Thought / Action / Observation panels."""

    def __init__(self, bus: TraceBus, *, verbose: bool = False) -> None:
        """Attach a listener to the bus (imports Rich here so tests stay console-free)."""
        # Imported here so TraceBus (and tests / a future SSE frontend) have
        # no console dependency at import time.
        from rich.console import Console, Group
        from rich.live import Live
        from rich.markdown import Markdown
        from rich.padding import Padding
        from rich.panel import Panel
        from rich.spinner import Spinner
        from rich.syntax import Syntax
        from rich.text import Text
        from rich.tree import Tree

        self._Console = Console
        self._Group = Group
        self._Live = Live
        self._Markdown = Markdown
        self._Padding = Padding
        self._Panel = Panel
        self._Spinner = Spinner
        self._Syntax = Syntax
        self._Text = Text
        self._Tree = Tree
        self.bus = bus
        self.verbose = verbose
        self.console = Console()
        self._live: Live | None = None
        bus.subscribe(self)

    def on_start(self, bus: TraceBus) -> None:
        """Print the header panel and start the live agent tree."""
        goal_preview = bus.goal.strip() or "(no goal)"
        header = self._Group(
            self._Text.from_markup(
                f"[bold]model[/] {bus.config.get('model')}   "
                f"[bold]host[/] {bus.config.get('host')}   "
                f"[bold]temp[/] {bus.config.get('temperature')}"
            ),
            self._Text(goal_preview, style="italic"),
        )
        workflow = str(bus.config.get("workflow") or "research")
        self.console.print(
            self._Panel(
                header,
                title=f"Dossier · {workflow}",
                border_style="bright_blue",
                padding=(0, 1),
            )
        )
        self._live = self._Live(
            self._status_renderable(),
            console=self.console,
            refresh_per_second=10,
            transient=False,
        )
        self._live.start()

    def on_stop(self, bus: TraceBus) -> None:
        """Stop the live view."""
        del bus
        if self._live is not None:
            self._live.stop()
            self._live = None

    def on_tree(self, bus: TraceBus) -> None:
        """Refresh the live agent tree."""
        del bus
        if self._live is not None:
            self._live.update(self._status_renderable())

    def on_event(self, bus: TraceBus, event: TraceEvent) -> None:
        """Render one event below the live tree (depth-indented when verbose)."""
        renderable = self._render_event(bus, event)
        if renderable is None:
            return
        out = renderable
        if self.verbose:
            depth = bus.depth(event.agent_id)
            if depth:
                out = self._Padding(renderable, (0, 0, 0, depth * 2))
        self._print(out)

    def on_complete(self, bus: TraceBus, citation_summary: str = "") -> None:
        """End of run: citation line, final-answer panel (or graceful exit), report path."""
        self.on_stop(bus)
        if citation_summary:
            if "unverified:" in citation_summary:
                style = "yellow"
            elif "none cited" in citation_summary:
                style = "dim"
            else:
                style = "green"
            self.console.print(self._Text(f"[{style}]{citation_summary}[/{style}]"))
        if bus.final:
            self.console.print()
            self.console.print(
                self._Panel(
                    self._Markdown(bus.final),
                    title="Final answer",
                    border_style="green",
                    padding=(1, 2),
                )
            )
        elif bus.reason:
            self.console.print(
                self._Panel(
                    f"Stopped without final_answer ([bold]{bus.reason}[/]).",
                    title="Graceful exit",
                    border_style="yellow",
                )
            )
        if bus.report_path:
            self.console.print(f"[dim]Saved report:[/] {bus.report_path}")

    def _render_event(self, bus: TraceBus, event: TraceEvent):
        """One TraceEvent → Rich renderable (or None when hidden at this verbosity)."""
        color = _role_color(event.role)
        if event.kind == "thought":
            if not self.verbose:
                return None
            body = self._Text(_preview(event.text, OBS_TRUNCATE, True) or "(no explicit thought)")
            body.stylize("italic dim")
            return self._Panel(
                body,
                title=f"[{color}]Thought[/{color}] · {event.agent_id} · step {event.step}",
                border_style=color,
                padding=(0, 1),
            )
        if event.kind == "action":
            if not self.verbose:
                detail = _one_line(_tool_detail(event.tool, event.args))
                return self._Text.assemble(
                    ("  " * bus.depth(event.agent_id)),
                    (f"{event.agent_id}", color),
                    (f"  {event.tool}  ", "bold yellow"),
                    (detail, "dim"),
                )
            title = self._Text.assemble(
                ("Action", f"bold {color}"),
                (f" · {event.agent_id} · ", "dim"),
                (event.tool, "bold yellow"),
            )
            blob = json.dumps(event.args, indent=2, ensure_ascii=False, default=str)
            return self._Panel(
                self._Syntax(blob, "json", theme="monokai", word_wrap=True, padding=0),
                title=title,
                border_style="yellow",
                padding=(0, 1),
            )
        if event.kind == "observation":
            if not self.verbose:
                summary = _observation_summary(event.text)
                if summary.startswith("evaluator ") or summary.startswith("page blocked"):
                    return self._Text.assemble(
                        ("  " * bus.depth(event.agent_id)),
                        (f"{event.agent_id}  ", color),
                        (summary, "green" if "PASS" in summary else "yellow"),
                    )
                return None
            preview = _preview(event.text, OBS_TRUNCATE, True)
            return self._Panel(
                self._Text(preview or "(empty)"),
                title=f"[{color}]Observation[/{color}] · {event.agent_id} · step {event.step}",
                border_style="dim",
                padding=(0, 1),
            )
        if event.kind == "note":
            show = self.verbose or event.text.startswith("Forcing a final_answer") or event.text.startswith(
                "Fallback ["
            ) or event.text.startswith("Writing final_answer")
            if not show:
                return None
            return self._Text.from_markup(f"[yellow]{event.agent_id}[/] {event.text}")
        return None

    def _print(self, renderable) -> None:
        """Print below the live view (or to the console when it is not running)."""
        console = self._live.console if self._live is not None else self.console
        console.print(renderable)

    def _status_renderable(self):
        """The live 'agent tree' panel (elapsed time + per-agent status)."""
        elapsed = max((datetime.now().astimezone() - self.bus.started).total_seconds(), 0.0)
        title = f"{self.bus.config.get('model', 'ollama')}  ·  {elapsed:.0f}s"
        tree = self._Tree(self._Text(title, style="bold bright_blue"))
        if not self.bus.roots:
            tree.add(self._Text("starting…", style="dim"))
        for root in self.bus.roots:
            self._add_node(tree, root)
        return self._Panel(tree, title="Live agents", border_style="bright_blue", padding=(0, 1))

    def _add_node(self, tree, node: AgentNode) -> None:
        """Recursively render one agent + children (spinner while busy)."""
        color = _role_color(node.role)
        cap = f"{node.step}/{node.max_iterations}" if node.max_iterations else str(node.step or "–")
        detail = node.activity or node.status
        label_text = f"{node.agent_id}  step {cap}  {detail}".rstrip()
        if node.busy:
            label = self._Spinner("dots", text=self._Text(label_text, style=color))
        elif node.status == "done":
            label = self._Text.from_markup(f"[green]✓[/] [{color}]{label_text}[/{color}]")
        else:
            label = self._Text(label_text, style=color)
        branch = tree.add(label)
        for child in node.children:
            self._add_node(branch, child)


class TracePrinter:
    """Per-agent adapter used by the ReAct loop."""

    def __init__(
        self,
        role: str,
        max_iterations: int = 0,
        parent_id: str | None = None,
    ) -> None:
        """Claim an agent node on the bus; constructed by run_react and the evaluator."""
        self.role = role
        self.session = get_bus()
        self.agent_id = self.session.start_agent(
            role, max_iterations=max_iterations, parent_id=parent_id
        )
        self.step = 0

    def next_step(self) -> int:
        """Advance to the next ReAct step."""
        self.step += 1
        self.session.set_step(self.agent_id, self.step)
        return self.step

    def thinking(self):
        """Bus span marking this agent as thinking."""
        return self.session.thinking(self.agent_id)

    def acting(self, tool: str, args: dict[str, Any]):
        """Bus span marking this agent as running a tool."""
        return self.session.acting(self.agent_id, tool, args)

    def thought(self, text: str) -> None:
        """Emit the step's Thought (reasoning + visible text)."""
        self.session.thought(self.agent_id, self.role, self.step, text.strip() or "(no explicit thought)")

    def action(self, name: str, args: Any) -> None:
        """Emit the step's Action (tool + args)."""
        payload = args if isinstance(args, dict) else {"value": args}
        self.session.action(self.agent_id, self.role, self.step, name, payload)

    def observation(self, text: str) -> None:
        """Emit the step's Observation (tool output)."""
        self.session.observation(self.agent_id, self.role, self.step, text)

    def note(self, text: str) -> None:
        """Emit a diagnostic Note."""
        self.session.note(self.agent_id, self.role, self.step, text)

    def finish(self, reason: str) -> None:
        """Close the agent's node with its stop reason."""
        self.session.end_agent(self.agent_id, reason)


def try_get_bus() -> TraceBus | None:
    """The active run's bus, or None if no start_trace is in progress."""
    return _current.get()


def get_bus() -> TraceBus:
    """Active run bus, or a process-wide fallback so tracing never crashes."""
    session = try_get_bus()
    if session is not None:
        return session
    global _fallback
    if _fallback is None:
        _fallback = TraceBus(save=False)
    return _fallback


def current_agent() -> tuple[str, str]:
    """Return (agent_id, role) for the agent on this thread, or empty strings."""
    stack = _agent_stack.get()
    if not stack:
        return "", ""
    agent_id = stack[-1]
    role = agent_id.rsplit("-", 1)[0] if "-" in agent_id else agent_id
    return agent_id, role


@contextmanager
def start_trace(
    *,
    goal: str,
    verbose: bool = False,
    save: bool = True,
    report_dir: str | Path = REPORT_DIR,
    config: dict[str, Any] | None = None,
    workflow: str = "",
    render: bool = True,
    browser: bool = True,
) -> Iterator[TraceBus]:
    """Own one run: counters + bus (+ renderer, browser pool), torn down on exit."""
    from workflow.tools.web import BrowserPool, set_browser_pool

    counters = Counters()
    counters_token = install_counters(counters)
    merged = dict(config or {})
    if workflow:
        merged["workflow"] = workflow
    bus = TraceBus(
        goal=goal,
        verbose=verbose,
        save=save,
        report_dir=report_dir,
        config=merged or None,
    )
    if render:
        RichRenderer(bus, verbose=verbose)
    token = _current.set(bus)
    pool: BrowserPool | None = None
    pool_token = None
    bus.start()
    if browser:
        pool = BrowserPool()
        try:
            pool.start()
            pool_token = set_browser_pool(pool)
        except Exception as exc:
            logger.warning("Browser pool unavailable, falling back to per-call launch: %s", exc)
            pool = None
    try:
        yield bus
    except Exception:
        bus.reason = bus.reason or "error"
        bus.complete(reason=bus.reason or "error")
        raise
    finally:
        if pool is not None:
            pool.close()
        if pool_token is not None:
            _browser_pool_reset(pool_token)
        bus.stop()
        _current.reset(token)
        uninstall_counters(counters_token)


def _browser_pool_reset(token) -> None:
    """Undo the run's shared BrowserPool ContextVar binding (start_trace teardown)."""
    from workflow.tools.web import _browser_pool

    try:
        _browser_pool.reset(token)
    except Exception:
        from workflow.tools.web import set_browser_pool

        set_browser_pool(None)
