"""Rich live agent tree + Thought / Action / Observation panels."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.text import Text
from rich.tree import Tree

from workflow.agents.report import TraceEvent, write_reports
from workflow.config import HOST, MODEL, NUM_PREDICT, OBS_TRUNCATE, REPORT_DIR, TEMPERATURE

_current: ContextVar[TraceSession | None] = ContextVar("trace_session", default=None)
_agent_stack: ContextVar[tuple[str, ...]] = ContextVar("agent_stack", default=())
_fallback: TraceSession | None = None

_ROLE_COLOR = {
    "planner": "cyan",
    "researcher": "magenta",
    "evaluator": "green",
}


def _role_color(role: str) -> str:
    return _ROLE_COLOR.get(role, "white")


def _preview(text: str, limit: int, verbose: bool) -> str:
    text = text or ""
    if verbose or len(text) <= limit:
        return text
    hidden = len(text) - limit
    return text[:limit].rstrip() + f"\n… [{hidden} more chars — full text in the run report]"


def _tool_detail(name: str, args: dict[str, Any]) -> str:
    for key in ("query", "url", "task", "expression"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            short = value.strip().replace("\n", " ")
            if len(short) > 72:
                short = short[:69] + "…"
            return f"{name} · {short}"
    return name


def _args_renderable(args: dict[str, Any]) -> RenderableType:
    blob = json.dumps(args, indent=2, ensure_ascii=False, default=str)
    return Syntax(blob, "json", theme="monokai", word_wrap=True, padding=0)


@dataclass
class AgentNode:
    agent_id: str
    role: str
    max_iterations: int = 0
    step: int = 0
    status: str = "idle"
    activity: str = ""
    children: list[AgentNode] = field(default_factory=list)

    @property
    def busy(self) -> bool:
        return self.status in {"thinking", "acting", "waiting"}


class TraceSession:
    """One run: live tree at the bottom, pretty panels in scrollback, saved report."""

    def __init__(
        self,
        *,
        goal: str = "",
        verbose: bool = False,
        save: bool = True,
        report_dir: str | Path = REPORT_DIR,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.goal = goal
        self.verbose = verbose
        self.save = save
        self.report_dir = Path(report_dir)
        self.config = config or {
            "model": MODEL,
            "host": HOST,
            "temperature": TEMPERATURE,
            "num_predict": NUM_PREDICT,
        }
        self.console = Console()
        self.events: list[TraceEvent] = []
        self.roots: list[AgentNode] = []
        self.nodes: dict[str, AgentNode] = {}
        self._role_counts: dict[str, int] = {}
        self._lock = threading.RLock()
        self.started = datetime.now().astimezone()
        self.final = ""
        self.reason = ""
        self.report_path: Path | None = None
        self._live: Live | None = None

    def start(self) -> None:
        goal_preview = self.goal.strip() or "(no goal)"
        header = Group(
            Text.from_markup(
                f"[bold]model[/] {self.config.get('model')}   "
                f"[bold]host[/] {self.config.get('host')}   "
                f"[bold]temp[/] {self.config.get('temperature')}"
            ),
            Text(goal_preview, style="italic"),
        )
        self.console.print(
            Panel(header, title="Agentic workflow", border_style="bright_blue", padding=(0, 1))
        )
        self._live = Live(
            self._status_renderable(),
            console=self.console,
            refresh_per_second=10,
            transient=False,
        )
        self._live.start()

    def stop(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def complete(self, *, final: str = "", reason: str = "") -> None:
        self.final = final or self.final
        self.reason = reason or self.reason
        self._refresh()
        self.stop()
        if self.final:
            self.console.print()
            self.console.print(
                Panel(
                    Markdown(self.final),
                    title="Final answer",
                    border_style="green",
                    padding=(1, 2),
                )
            )
        elif self.reason:
            self.console.print(
                Panel(
                    f"Stopped without final_answer ([bold]{self.reason}[/]).",
                    title="Graceful exit",
                    border_style="yellow",
                )
            )
        if self.save:
            self._write_reports()
            if self.report_path:
                self.console.print(f"[dim]Saved report:[/] {self.report_path}")

    def start_agent(self, role: str, max_iterations: int = 0) -> str:
        with self._lock:
            n = self._role_counts.get(role, 0) + 1
            self._role_counts[role] = n
            agent_id = f"{role}-{n}"
            node = AgentNode(agent_id=agent_id, role=role, max_iterations=max_iterations)
            stack = list(_agent_stack.get())
            parent_id = stack[-1] if stack else None
            parent = self.nodes.get(parent_id) if parent_id else None
            if parent:
                parent.children.append(node)
                parent.status = "waiting"
                parent.activity = f"waiting on {role}"
            else:
                self.roots.append(node)
            stack.append(agent_id)
            _agent_stack.set(tuple(stack))
            self.nodes[agent_id] = node
        self._emit(
            TraceEvent(kind="spawn", role=role, agent_id=agent_id, step=0, text=role),
            print_panel=False,
        )
        self._refresh()
        return agent_id

    def end_agent(self, agent_id: str, reason: str) -> None:
        with self._lock:
            node = self.nodes.get(agent_id)
            if node:
                node.status = "done"
                node.activity = reason
            stack = list(_agent_stack.get())
            if stack and stack[-1] == agent_id:
                stack.pop()
                _agent_stack.set(tuple(stack))
            parent_id = stack[-1] if stack else None
            parent = self.nodes.get(parent_id) if parent_id else None
            if parent and parent.status == "waiting":
                still_busy = any(child.busy for child in parent.children)
                if not still_busy:
                    parent.status = "acting"
                    parent.activity = parent.activity.replace("waiting on ", "resuming after ")
        self._emit(
            TraceEvent(kind="finish", role=node.role if node else "", agent_id=agent_id, step=0, text=reason),
            print_panel=False,
        )
        self._refresh()

    def set_step(self, agent_id: str, step: int) -> None:
        with self._lock:
            node = self.nodes.get(agent_id)
            if node:
                node.step = step
        self._refresh()

    @contextmanager
    def thinking(self, agent_id: str) -> Iterator[None]:
        node = self.nodes.get(agent_id)
        if node:
            node.status = "thinking"
            node.activity = "waiting on Ollama"
            self._refresh()
        try:
            yield
        finally:
            if node and node.status == "thinking":
                node.status = "idle"
                node.activity = ""
                self._refresh()

    @contextmanager
    def acting(self, agent_id: str, tool: str, args: dict[str, Any]) -> Iterator[None]:
        node = self.nodes.get(agent_id)
        if node:
            node.status = "acting"
            node.activity = _tool_detail(tool, args)
            self._refresh()
        try:
            yield
        finally:
            if node and node.status == "acting":
                node.status = "idle"
                self._refresh()

    def thought(self, agent_id: str, role: str, step: int, text: str) -> None:
        event = TraceEvent(kind="thought", role=role, agent_id=agent_id, step=step, text=text)
        color = _role_color(role)
        body = Text(_preview(text, OBS_TRUNCATE, self.verbose) or "(no explicit thought)")
        body.stylize("italic dim")
        self._emit(
            event,
            Panel(
                body,
                title=f"[{color}]Thought[/{color}] · {agent_id} · step {step}",
                border_style=color,
                padding=(0, 1),
            ),
            depth=self._depth(agent_id),
        )

    def action(self, agent_id: str, role: str, step: int, name: str, args: dict[str, Any]) -> None:
        event = TraceEvent(
            kind="action", role=role, agent_id=agent_id, step=step, tool=name, args=args
        )
        color = _role_color(role)
        title = Text.assemble(
            ("Action", f"bold {color}"),
            (f" · {agent_id} · ", "dim"),
            (name, "bold yellow"),
        )
        self._emit(
            event,
            Panel(
                _args_renderable(args),
                title=title,
                border_style="yellow",
                padding=(0, 1),
            ),
            depth=self._depth(agent_id),
        )

    def observation(self, agent_id: str, role: str, step: int, text: str) -> None:
        event = TraceEvent(kind="observation", role=role, agent_id=agent_id, step=step, text=text)
        color = _role_color(role)
        preview = _preview(text, OBS_TRUNCATE, self.verbose)
        self._emit(
            event,
            Panel(
                Text(preview or "(empty)"),
                title=f"[{color}]Observation[/{color}] · {agent_id} · step {step}",
                border_style="dim",
                padding=(0, 1),
            ),
            depth=self._depth(agent_id),
        )

    def note(self, agent_id: str, role: str, step: int, text: str) -> None:
        event = TraceEvent(kind="note", role=role, agent_id=agent_id, step=step, text=text)
        self._emit(
            event,
            Text.from_markup(f"[yellow]{agent_id}[/] {text}"),
            depth=self._depth(agent_id),
        )

    def _depth(self, agent_id: str) -> int:
        node = self.nodes.get(agent_id)
        depth = 0
        while node:
            parent = self._parent_of(node)
            if parent is None:
                break
            depth += 1
            node = parent
        return depth

    def _parent_of(self, node: AgentNode) -> AgentNode | None:
        for candidate in self.nodes.values():
            if node in candidate.children:
                return candidate
        return None

    def _emit(
        self,
        event: TraceEvent,
        renderable: RenderableType | None = None,
        *,
        depth: int = 0,
        print_panel: bool = True,
    ) -> None:
        with self._lock:
            self.events.append(event)
            if print_panel and renderable is not None:
                out: RenderableType = renderable
                if depth:
                    out = Padding(renderable, (0, 0, 0, depth * 2))
                self._print(out)
            self._refresh()

    def _print(self, renderable: RenderableType) -> None:
        console = self._live.console if self._live is not None else self.console
        console.print(renderable)

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._status_renderable())

    def _status_renderable(self) -> RenderableType:
        elapsed = max((datetime.now().astimezone() - self.started).total_seconds(), 0.0)
        title = f"{self.config.get('model', 'ollama')}  ·  {elapsed:.0f}s"
        tree = Tree(Text(title, style="bold bright_blue"))
        if not self.roots:
            tree.add(Text("starting…", style="dim"))
        for root in self.roots:
            self._add_node(tree, root)
        return Panel(tree, title="Live agents", border_style="bright_blue", padding=(0, 1))

    def _add_node(self, tree: Tree, node: AgentNode) -> None:
        color = _role_color(node.role)
        cap = f"{node.step}/{node.max_iterations}" if node.max_iterations else str(node.step or "–")
        detail = node.activity or node.status
        label_text = f"{node.agent_id}  step {cap}  {detail}".rstrip()
        if node.busy:
            label: RenderableType = Spinner("dots", text=Text(label_text, style=color))
        elif node.status == "done":
            label = Text.from_markup(f"[green]✓[/] [{color}]{label_text}[/{color}]")
        else:
            label = Text(label_text, style=color)
        branch = tree.add(label)
        for child in node.children:
            self._add_node(branch, child)

    def _write_reports(self) -> None:
        ended = datetime.now().astimezone()
        stem = self.started.strftime("%Y%m%d-%H%M%S")
        self.report_path = write_reports(
            self.report_dir,
            stem,
            goal=self.goal,
            config=self.config,
            events=self.events,
            final=self.final,
            reason=self.reason,
            started=self.started,
            ended=ended,
        )


class TracePrinter:
    """Per-agent adapter used by the ReAct loop."""

    def __init__(self, role: str, indent: str = "", max_iterations: int = 0) -> None:
        self.role = role
        self.indent = indent
        self.session = get_session()
        self.agent_id = self.session.start_agent(role, max_iterations=max_iterations)
        self.step = 0

    def next_step(self) -> int:
        self.step += 1
        self.session.set_step(self.agent_id, self.step)
        return self.step

    def thinking(self):
        return self.session.thinking(self.agent_id)

    def acting(self, tool: str, args: dict[str, Any]):
        return self.session.acting(self.agent_id, tool, args)

    def thought(self, text: str) -> None:
        self.session.thought(self.agent_id, self.role, self.step, text.strip() or "(no explicit thought)")

    def action(self, name: str, args: Any) -> None:
        payload = args if isinstance(args, dict) else {"value": args}
        self.session.action(self.agent_id, self.role, self.step, name, payload)

    def observation(self, text: str) -> None:
        self.session.observation(self.agent_id, self.role, self.step, text)

    def note(self, text: str) -> None:
        self.session.note(self.agent_id, self.role, self.step, text)

    def finish(self, reason: str) -> None:
        self.session.end_agent(self.agent_id, reason)


def get_session() -> TraceSession:
    session = _current.get()
    if session is not None:
        return session
    global _fallback
    if _fallback is None:
        _fallback = TraceSession(save=False)
    return _fallback


@contextmanager
def start_trace(
    *,
    goal: str,
    verbose: bool = False,
    save: bool = True,
    report_dir: str | Path = REPORT_DIR,
    config: dict[str, Any] | None = None,
) -> Iterator[TraceSession]:
    session = TraceSession(
        goal=goal,
        verbose=verbose,
        save=save,
        report_dir=report_dir,
        config=config,
    )
    token = _current.set(session)
    session.start()
    try:
        yield session
    except Exception:
        session.reason = session.reason or "error"
        session.complete(reason=session.reason or "error")
        raise
    finally:
        session.stop()
        _current.reset(token)
