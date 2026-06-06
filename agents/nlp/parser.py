"""
NexusForge Advanced NLP Command Parser
Uses Anthropic Claude (via API) for richer natural-language swarm command interpretation.
Falls back to the keyword-based parser when Claude is unavailable.
"""

import json
import os
import re
from typing import Optional
from dataclasses import dataclass

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

from agents.swarm.orchestrator import (
    SwarmCommand, MissionType, Formation, TeamID,
    parse_nlp_command, _LOCATION_KEYWORDS,
)
from simulation.engine.sim import Vec2, ARENA_W, ARENA_H


SYSTEM_PROMPT = """You are the tactical AI for NexusForge, a drone swarm combat arena.
Parse natural-language swarm commands and return structured JSON.

Available teams: RED, BLUE, GREEN, GOLD
Available missions: attack, defend, capture, flank, regroup, scatter, surround, patrol, escort, kamikaze
Available formations: wedge, line, circle, spread, diamond, column
Named locations: center, north, south, east, west, nexus, alpha_point (200,450), beta_point (600,200), gamma_point (600,700), delta_point (1000,450)
Arena size: 1200 x 900

Respond ONLY with valid JSON in this exact format:
{
  "team": "RED",
  "mission": "attack",
  "formation": "wedge",
  "target_position": {"x": 600, "y": 450},
  "target_team": null,
  "priority": 7,
  "reasoning": "one sentence explanation"
}

Rules:
- priority 1-10 (10 = most urgent)
- formation can be null
- target_position can be null
- target_team can be null if not targeting a specific team
- For "regroup" and "defend" always include a target_position
- For "kamikaze" set priority 10
"""


@dataclass
class ParsedCommand:
    team: TeamID
    mission: MissionType
    formation: Optional[Formation]
    target_position: Optional[Vec2]
    target_team: Optional[TeamID]
    priority: int
    reasoning: str
    source: str   # "claude" | "keyword" | "fallback"


def parse_with_claude(text: str, issuing_team: TeamID) -> Optional[ParsedCommand]:
    """
    Use Claude API for rich NLP command parsing.
    Returns None if Claude unavailable or request fails.
    """
    if not _ANTHROPIC_AVAILABLE:
        return None

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Issuing team: {issuing_team.name}\nCommand: {text}"
            }],
        )
        raw = message.content[0].text.strip()

        # Strip markdown fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        data = json.loads(raw)

        team = TeamID[data.get("team", issuing_team.name).upper()]
        mission = MissionType(data.get("mission", "patrol").lower())
        formation_raw = data.get("formation")
        formation = Formation(formation_raw.lower()) if formation_raw else None

        tp = data.get("target_position")
        target_pos = Vec2(float(tp["x"]), float(tp["y"])) if tp else None

        tt = data.get("target_team")
        target_team = TeamID[tt.upper()] if tt else None

        priority = max(1, min(10, int(data.get("priority", 5))))
        reasoning = data.get("reasoning", "")

        return ParsedCommand(
            team=team, mission=mission, formation=formation,
            target_position=target_pos, target_team=target_team,
            priority=priority, reasoning=reasoning, source="claude",
        )

    except Exception as e:
        # Claude unavailable or parse error — fall through to keyword parser
        return None


def parse_command(text: str, issuing_team: TeamID) -> Optional[SwarmCommand]:
    """
    Parse a natural-language command.
    Tries Claude API first, falls back to keyword parser.
    """
    # Try Claude first
    claude_result = parse_with_claude(text, issuing_team)
    if claude_result:
        return SwarmCommand(
            team=claude_result.team,
            mission=claude_result.mission,
            formation=claude_result.formation,
            target_position=claude_result.target_position,
            target_team=claude_result.target_team,
            priority=claude_result.priority,
            source="claude_nlp",
            raw_text=text,
        )

    # Fallback: keyword-based parser
    return parse_nlp_command(text, issuing_team)


def batch_parse(commands: list, issuing_team: TeamID) -> list:
    """Parse a list of commands and return structured results."""
    results = []
    for cmd_text in commands:
        cmd = parse_command(cmd_text, issuing_team)
        results.append({
            "input": cmd_text,
            "parsed": cmd.to_dict() if cmd else None,
            "success": cmd is not None,
        })
    return results


# ─── Demo / test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_commands = [
        "Red team, attack the center with wedge formation",
        "Defend the nexus, form a circle around it",
        "Flank the blue team from the east — high priority",
        "All units scatter and regroup at alpha point",
        "Surround the enemy at delta point",
        "Kamikaze run on the gold team",
        "Blue squadron, patrol the northern corridor in line formation",
    ]

    print("NexusForge NLP Parser Demo")
    print("=" * 50)
    for cmd_text in test_commands:
        result = parse_command(cmd_text, TeamID.RED)
        if result:
            print(f"  Input:    {cmd_text}")
            print(f"  Parsed:   {result.mission.value} | {result.formation} | "
                  f"pos={result.target_position} | source={result.source}")
            print()
        else:
            print(f"  FAILED:   {cmd_text}\n")
