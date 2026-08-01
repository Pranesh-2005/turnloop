"""Permission decisions.

Precedence is flat and evaluated in a fixed order. There is deliberately no
specificity scoring — "the most specific rule wins" is impossible to predict
once patterns overlap, and a permission system nobody can predict is a
permission system people disable.

    1. deny match                -> DENY, unconditionally, even in bypass mode
    2. plan mode + mutating tool  -> DENY (read-only enforcement)
    3. bypass mode                -> ALLOW
    4. allow match (incl. session grants) -> ALLOW
    5. ask match                  -> ASK
    6. mode default:
         default -> ASK for mutating tools, ALLOW for read-only
         auto    -> ALLOW
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from turnloop.config import PermissionConfig, PermissionMode
from turnloop.permissions.rules import Rule, parse_rules

if TYPE_CHECKING:
    from turnloop.tools.base import Tool


class Verdict(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass(slots=True)
class Decision:
    verdict: Verdict
    reason: str = ""
    rule: Rule | None = None

    @property
    def allowed(self) -> bool:
        return self.verdict is Verdict.ALLOW


@dataclass(slots=True)
class PermissionRequest:
    """Handed to the UI when a call needs a human answer."""

    tool_name: str
    target: str
    summary: str
    args_preview: str
    is_read_only: bool
    suggested_rule: str
    reason: str = ""
    subagent_id: str | None = None


class Scope(StrEnum):
    ONCE = "once"
    SESSION = "session"
    PROJECT = "project"  # persisted to .turnloop/settings.local.json


@dataclass(slots=True)
class PermissionDecision:
    """The human's answer."""

    approved: bool
    scope: Scope = Scope.ONCE
    rule: str | None = None
    reason: str = ""


@dataclass
class PermissionEngine:
    mode: PermissionMode = "default"
    allow: list[Rule] = field(default_factory=list)
    deny: list[Rule] = field(default_factory=list)
    ask: list[Rule] = field(default_factory=list)
    # Grants the human gave for this session only. Not persisted.
    session_grants: list[Rule] = field(default_factory=list)
    # Populated for auditing; the session log is the durable record.
    decisions: list[tuple[str, str, Verdict]] = field(default_factory=list)
    project_root: Path | None = None
    # The agent's working directory. Model-supplied relative paths resolve against
    # this, not against the interpreter's cwd.
    cwd: Path | None = None

    @classmethod
    def from_config(cls, cfg: PermissionConfig, mode: PermissionMode,
                    project_root: Path | None = None,
                    cwd: Path | None = None) -> PermissionEngine:
        return cls(
            mode=mode,
            allow=parse_rules(cfg.allow),
            deny=parse_rules(cfg.deny),
            ask=parse_rules(cfg.ask),
            project_root=project_root,
            cwd=cwd or project_root,
        )

    # --- the decision ------------------------------------------------------

    def check(self, tool: Tool, args: Any) -> Decision:
        target = tool.permission_target(args)
        read_only = tool.is_read_only_for(args)

        if rule := self._first_match(self.deny, tool, target):
            return self._record(
                tool, target,
                Decision(Verdict.DENY, f"denied by rule {rule}", rule),
            )

        if self.mode == "plan" and not read_only:
            return self._record(
                tool, target,
                Decision(
                    Verdict.DENY,
                    "plan mode is read-only: propose the change instead of making it",
                ),
            )

        if self.mode == "bypass":
            return self._record(tool, target, Decision(Verdict.ALLOW, "bypass mode"))

        if rule := self._first_match(self.allow + self.session_grants, tool, target):
            return self._record(tool, target, Decision(Verdict.ALLOW, f"allowed by {rule}", rule))

        if rule := self._first_match(self.ask, tool, target):
            return self._record(
                tool, target, Decision(Verdict.ASK, f"confirmation required by {rule}", rule)
            )

        if self.mode == "auto":
            return self._record(tool, target, Decision(Verdict.ALLOW, "auto mode"))

        if read_only:
            return self._record(tool, target, Decision(Verdict.ALLOW, "read-only tool"))

        return self._record(
            tool, target, Decision(Verdict.ASK, "modifies state and no rule covers it")
        )

    def _first_match(self, rules: list[Rule], tool: Tool, target: str) -> Rule | None:
        for rule in rules:
            if not rule.tool_matches(tool.name):
                continue
            if rule.pattern is None:
                return rule
            if tool.match_target(target, rule.pattern, self.project_root, self.cwd):
                return rule
        return None

    def _record(self, tool: Tool, target: str, decision: Decision) -> Decision:
        self.decisions.append((tool.name, target, decision.verdict))
        return decision

    # --- grants ------------------------------------------------------------

    def grant(self, rule_text: str, scope: Scope = Scope.SESSION) -> Rule:
        from turnloop.permissions.rules import parse_rule

        rule = parse_rule(rule_text, source=scope.value)
        if scope is Scope.SESSION:
            self.session_grants.append(rule)
        elif scope is Scope.PROJECT:
            self.allow.append(rule)
        return rule

    def suggested_rule(self, tool: Tool, args: Any) -> str:
        """The rule text offered as "always allow this".

        Deliberately narrow. For Bash it suggests the first two words plus a
        wildcard (`git commit *`), never a bare `Bash`, because a broad grant
        made in a hurry is how permission systems stop meaning anything.
        """
        target = tool.permission_target(args)
        if tool.name == "Bash":
            from turnloop.permissions.rules import normalize_command, split_shell_command

            segments = split_shell_command(target).segments
            first = normalize_command(segments[0] if segments else target)
            words = first.split()
            head = " ".join(words[:2]) if len(words) > 1 else (words[0] if words else target)
            return f"Bash({head} *)"
        if target == tool.name:
            return tool.name
        try:
            rel = Path(target)
            if self.project_root and rel.is_absolute():
                rel = rel.resolve().relative_to(self.project_root.resolve())
            parent = rel.parent.as_posix()
            if parent in ("", "."):
                return f"{tool.name}({rel.name})"
            return f"{tool.name}({parent}/**)"
        except ValueError:
            return f"{tool.name}({target})"

    def snapshot(self) -> dict:
        return {
            "mode": self.mode,
            "allow": [str(r) for r in self.allow],
            "deny": [str(r) for r in self.deny],
            "ask": [str(r) for r in self.ask],
            "session_grants": [str(r) for r in self.session_grants],
        }
