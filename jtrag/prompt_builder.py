from __future__ import annotations

from typing import Dict, List, Sequence

from .anes_io import EntityRecord, GraphSnapshot, SampleRecord


SYSTEM_PROMPT = (
    "You are a biomedical relation prediction assistant. "
    "Follow the instruction exactly and answer only Yes or No."
)


def format_chat_prompt(tokenizer, prompt: str) -> str:

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM_PROMPT}\n\n{prompt}\nAnswer:"


def _required_entity_name(entity: EntityRecord) -> str:

    name = (entity.name or "").replace("\n", " ").strip()
    if not name:
        raise ValueError(f"实体 eid={entity.eid} 缺少真实名称，无法构造仅使用实体名称的 Prompt。")
    return name


def relation_side(graph: GraphSnapshot, src: int, dst: int, v: int) -> tuple[str, str, str]:

    e_src = graph.edge(src, v)
    e_dst = graph.edge(dst, v)
    if e_src is not None and e_dst is not None:
        side = "both Entity A and Entity B"
    elif e_src is not None:
        side = "Entity A"
    elif e_dst is not None:
        side = "Entity B"
    else:
        side = "unknown"

    first_times: List[str] = []
    frequencies: List[str] = []
    if e_src is not None:
        first_times.append(f"A:{e_src.first_time}")
        frequencies.append(f"A:{e_src.freq}")
    if e_dst is not None:
        first_times.append(f"B:{e_dst.first_time}")
        frequencies.append(f"B:{e_dst.freq}")
    return (
        side,
        ", ".join(first_times) if first_times else "unknown",
        ", ".join(frequencies) if frequencies else "unknown",
    )


def build_prompt(
    sample: SampleRecord,
    selected_eids: Sequence[int],
    entities: Dict[int, EntityRecord],
    graph: GraphSnapshot,
) -> str:

    src_ent = entities[int(sample.src)]
    dst_ent = entities[int(sample.dst)]
    src_name = _required_entity_name(src_ent)
    dst_name = _required_entity_name(dst_ent)

    lines: List[str] = [
        "Use only the historical neighbor evidence below to predict whether the two query entities will have a relation at any future time.",
        "Do not provide explanations. Answer only Yes or No.",
        "",
        "[Query Entities]",
        f"Entity A: {src_name}",
        f"Entity B: {dst_name}",
        "",
        "[Historical Neighbor Evidence]",
    ]

    if not selected_eids:
        lines.append("No sufficiently relevant historical neighbor evidence was selected.")
    else:
        for rank, eid in enumerate(selected_eids, start=1):
            ent = entities[int(eid)]
            neighbor_name = _required_entity_name(ent)
            side, first_times, frequencies = relation_side(
                graph, int(sample.src), int(sample.dst), int(eid)
            )
            lines.append(f"{rank}. Neighbor: {neighbor_name}")
            lines.append(f"   Connected to: {side}")
            lines.append(f"   Historical first co-occurrence: {first_times}")
            lines.append(f"   Historical co-occurrence count: {frequencies}")

    lines.extend(
        [
            "",
            "[Question]",
            "Will Entity A and Entity B have a relation at any future time?",
            "Answer only Yes or No.",
        ]
    )
    return "\n".join(lines)
