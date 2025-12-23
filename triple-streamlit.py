import ast
import io
import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st


# --------------------------
# Helpers
# --------------------------

UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)

def _safe_parse_edge_uuid_list(x: Any) -> List[str]:
    """
    Parse the episodic 'entity_edges' field into a list of UUID strings.
    Handles common formats:
      - Python list string: "['uuid1','uuid2']"
      - JSON list string: '["uuid1","uuid2"]'
      - comma-separated: "uuid1, uuid2"
      - already-list
      - empty / NaN
      - messy text containing UUIDs
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []

    if isinstance(x, list):
        # keep only uuid-like strings
        uuids = []
        for v in x:
            if isinstance(v, str):
                uuids.extend(UUID_RE.findall(v))
        return list(dict.fromkeys(uuids))

    if not isinstance(x, str):
        x = str(x)

    s = x.strip()
    if not s:
        return []

    # Fast path: extract UUIDs from any string
    found = UUID_RE.findall(s)
    if found:
        return list(dict.fromkeys(found))

    # Try literal_eval for list-like
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            uuids = []
            for item in v:
                if isinstance(item, str):
                    uuids.extend(UUID_RE.findall(item))
            return list(dict.fromkeys(uuids))
    except Exception:
        pass

    # Fallback: split by comma
    parts = [p.strip() for p in s.split(",") if p.strip()]
    uuids = []
    for p in parts:
        uuids.extend(UUID_RE.findall(p))
    return list(dict.fromkeys(uuids))


def _guess_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def _ensure_required_cols(df: pd.DataFrame, required: List[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}. Found columns: {list(df.columns)}")


def _build_entity_lookup(nodes_entity: pd.DataFrame) -> Tuple[Dict[int, str], str]:
    """
    Returns: id->name map, and the column used for name.
    """
    id_col = _guess_col(nodes_entity, ["id", "node_id"])
    if not id_col:
        raise ValueError("nodes_Entity.csv must have an 'id' column (or 'node_id').")

    # prefer common columns
    name_col = _guess_col(nodes_entity, ["name", "label", "title", "value"])
    if not name_col:
        # fallback to first non-id column
        non_id = [c for c in nodes_entity.columns if c != id_col]
        if not non_id:
            raise ValueError("nodes_Entity.csv has no usable name/label column.")
        name_col = non_id[0]

    # build map
    m = {}
    for _, row in nodes_entity.iterrows():
        try:
            ent_id = int(row[id_col])
        except Exception:
            continue
        name = row.get(name_col, "")
        if pd.isna(name):
            name = ""
        m[ent_id] = str(name)
    return m, name_col


def _normalize_edges(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """
    Normalizes edge dfs to have:
      - uuid
      - from_id
      - to_id
      - edge_name (only for RELATES_TO; for MENTIONS we set edge_name='MENTIONS')
    """
    uuid_col = _guess_col(df, ["uuid", "id", "edge_id"])
    from_col = _guess_col(df, ["from_id", "source", "src", "from"])
    to_col = _guess_col(df, ["to_id", "target", "dst", "to"])
    name_col = _guess_col(df, ["edge_name", "label", "relation", "predicate", "type", "name"])

    if not uuid_col or not from_col or not to_col:
        raise ValueError(
            f"{kind} edges file must have uuid/id, from_id, to_id columns. "
            f"Found columns: {list(df.columns)}"
        )

    out = df.copy()
    out = out.rename(columns={uuid_col: "uuid", from_col: "from_id", to_col: "to_id"})
    out["uuid"] = out["uuid"].astype(str)

    if kind == "RELATES_TO":
        if not name_col:
            raise ValueError(
                "edges_RELATES_TO.csv must have an edge_name / relation / predicate column."
            )
        out = out.rename(columns={name_col: "edge_name"})
        out["edge_name"] = out["edge_name"].astype(str)
    else:
        out["edge_name"] = "MENTIONS"

    # ensure numeric ids where possible
    for c in ["from_id", "to_id"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    return out[["uuid", "from_id", "edge_name", "to_id"]]


def _build_episode_triples(
    nodes_ep: pd.DataFrame,
    entity_name_by_id: Dict[int, str],
    edges_relates: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - episode_rows: one row per episodic node with 'episodic_id', 'sent', 'edge_uuids', 'n_rel_edges', 'triples'
      - rel_edges_expanded: (episodic_id, uuid, from_id, edge_name, to_id, sub_name, obj_name)
    """
    ep_id_col = _guess_col(nodes_ep, ["id", "node_id"])
    content_col = _guess_col(nodes_ep, ["content", "text", "sent", "sentence"])
    edges_col = _guess_col(nodes_ep, ["entity_edges", "edges", "edge_uuids"])

    if not ep_id_col:
        raise ValueError("nodes_Episodic.csv must have an 'id' column (or 'node_id').")
    if not content_col:
        raise ValueError("nodes_Episodic.csv must have a 'content'/'text' column for the chunk text.")
    if not edges_col:
        raise ValueError("nodes_Episodic.csv must have an 'entity_edges' column (or similar) that lists edge UUIDs.")

    # Build a lookup: uuid -> edge row for RELATES_TO
    rel_by_uuid = edges_relates.set_index("uuid", drop=False)

    episode_out_rows = []
    expanded_rows = []

    for _, row in nodes_ep.iterrows():
        try:
            episodic_id = int(row[ep_id_col])
        except Exception:
            continue

        sent = row.get(content_col, "")
        if pd.isna(sent):
            sent = ""
        sent = str(sent)

        edge_uuids = _safe_parse_edge_uuid_list(row.get(edges_col, None))

        # Keep only those uuids that exist in RELATES_TO
        rel_uuids = [u for u in edge_uuids if u in rel_by_uuid.index]

        triples = []
        for u in rel_uuids:
            e = rel_by_uuid.loc[u]
            from_id = e["from_id"]
            to_id = e["to_id"]
            pred = str(e["edge_name"])

            sub_name = entity_name_by_id.get(int(from_id)) if pd.notna(from_id) else None
            obj_name = entity_name_by_id.get(int(to_id)) if pd.notna(to_id) else None

            sub_name = sub_name if sub_name is not None else f"[MISSING_ENTITY:{from_id}]"
            obj_name = obj_name if obj_name is not None else f"[MISSING_ENTITY:{to_id}]"

            triples.append([sub_name, pred, obj_name])

            expanded_rows.append(
                {
                    "episodic_id": episodic_id,
                    "uuid": u,
                    "from_id": from_id,
                    "edge_name": pred,
                    "to_id": to_id,
                    "sub_name": sub_name,
                    "obj_name": obj_name,
                }
            )

        episode_out_rows.append(
            {
                "episodic_id": episodic_id,
                "sent": sent,
                "edge_uuids_total": len(edge_uuids),
                "rel_uuids": rel_uuids,
                "n_rel_edges": len(rel_uuids),
                "triples": triples,  # list[list[str,str,str]]
            }
        )

    episode_df = pd.DataFrame(episode_out_rows).sort_values("episodic_id").reset_index(drop=True)
    expanded_df = pd.DataFrame(expanded_rows)
    if not expanded_df.empty:
        expanded_df = expanded_df.sort_values(["episodic_id", "edge_name", "uuid"]).reset_index(drop=True)
    return episode_df, expanded_df


def _attach_mentions(
    episode_df: pd.DataFrame,
    nodes_ep: pd.DataFrame,
    edges_mentions: pd.DataFrame,
    entity_name_by_id: Dict[int, str],
) -> pd.DataFrame:
    """
    Adds mention_entities (list of entity names) and n_mentions.
    Assumption: edges_MENTIONS is episodic -> entity.
    """
    if edges_mentions is None or edges_mentions.empty:
        episode_df["n_mentions"] = 0
        episode_df["mention_entities"] = [[] for _ in range(len(episode_df))]
        return episode_df

    # Group mentions by episodic from_id
    # Some pipelines store from_id=episodic, to_id=entity; if reversed, we detect.
    # Heuristic: episodic ids are those that appear in episode_df. We'll see which side matches more.
    ep_ids = set(episode_df["episodic_id"].astype(int).tolist())

    from_matches = edges_mentions["from_id"].dropna().astype(int).isin(ep_ids).sum()
    to_matches = edges_mentions["to_id"].dropna().astype(int).isin(ep_ids).sum()

    if to_matches > from_matches:
        # swap
        m = edges_mentions.rename(columns={"from_id": "to_id", "to_id": "from_id"}).copy()
    else:
        m = edges_mentions.copy()

    grouped = {}
    for _, r in m.iterrows():
        if pd.isna(r["from_id"]) or pd.isna(r["to_id"]):
            continue
        ep = int(r["from_id"])
        ent = int(r["to_id"])
        name = entity_name_by_id.get(ent, f"[MISSING_ENTITY:{ent}]")
        grouped.setdefault(ep, []).append(name)

    mention_entities = []
    mention_counts = []
    for ep in episode_df["episodic_id"].astype(int).tolist():
        ents = grouped.get(ep, [])
        # unique but keep order
        seen = set()
        uniq = []
        for e in ents:
            if e not in seen:
                seen.add(e)
                uniq.append(e)
        mention_entities.append(uniq)
        mention_counts.append(len(uniq))

    episode_df = episode_df.copy()
    episode_df["n_mentions"] = mention_counts
    episode_df["mention_entities"] = mention_entities
    return episode_df


def _jsonl_bytes(episode_df: pd.DataFrame, id_style: str) -> bytes:
    """
    Creates evaluator-friendly JSONL:
      {"id": <...>, "sent": <...>, "triples": [[sub, rel, obj], ...]}
    id_style:
      - "episodic_id" uses the original episodic node id as the JSONL id
      - "row_index" uses 0..N-1
      - "prefixed" uses "episodic_<id>"
    """
    lines = []
    for i, row in episode_df.reset_index(drop=True).iterrows():
        if id_style == "row_index":
            rid = str(i)
        elif id_style == "prefixed":
            rid = f"episodic_{int(row['episodic_id'])}"
        else:
            rid = str(int(row["episodic_id"]))

        obj = {
            "id": rid,
            "sent": row["sent"],
            "triples": row["triples"],
        }
        lines.append(json.dumps(obj, ensure_ascii=False))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _csv_bytes(episode_df: pd.DataFrame) -> bytes:
    out = episode_df.copy()
    # store triples as JSON strings for CSV readability
    out["triples"] = out["triples"].apply(lambda x: json.dumps(x, ensure_ascii=False))
    out["mention_entities"] = out.get("mention_entities", [[] for _ in range(len(out))]).apply(
        lambda x: json.dumps(x, ensure_ascii=False)
    )
    return out.to_csv(index=False).encode("utf-8")


# --------------------------
# UI
# --------------------------

st.set_page_config(page_title="Triples per Chunk Explorer", layout="wide")
st.title("Triples per Chunk Explorer (Episodic → RELATES_TO → Entity names)")

with st.sidebar:
    st.header("1) Upload your CSVs")
    f_nodes_ep = st.file_uploader("nodes_Episodic.csv", type=["csv"])
    f_nodes_ent = st.file_uploader("nodes_Entity.csv", type=["csv"])
    f_edges_rel = st.file_uploader("edges_RELATES_TO.csv", type=["csv"])
    f_edges_men = st.file_uploader("edges_MENTIONS.csv (optional)", type=["csv"])

    st.divider()
    st.header("2) Export settings")
    id_style = st.selectbox(
        "JSONL id field style",
        options=["episodic_id", "prefixed", "row_index"],
        index=0,
        help="Choose how the JSONL 'id' will be written.",
    )
    show_only_with_triples = st.checkbox("Show only chunks with ≥1 RELATES_TO triple", value=False)

if not (f_nodes_ep and f_nodes_ent and f_edges_rel):
    st.info("Upload at least nodes_Episodic.csv, nodes_Entity.csv, and edges_RELATES_TO.csv to begin.")
    st.stop()

# Load data
try:
    nodes_ep = pd.read_csv(f_nodes_ep)
    nodes_ent = pd.read_csv(f_nodes_ent)
    edges_rel_raw = pd.read_csv(f_edges_rel)

    edges_men_raw = pd.read_csv(f_edges_men) if f_edges_men else None

    entity_name_by_id, entity_name_col = _build_entity_lookup(nodes_ent)
    edges_rel = _normalize_edges(edges_rel_raw, "RELATES_TO")
    edges_men = _normalize_edges(edges_men_raw, "MENTIONS") if edges_men_raw is not None else None

    episode_df, expanded_rel = _build_episode_triples(nodes_ep, entity_name_by_id, edges_rel)
    episode_df = _attach_mentions(episode_df, nodes_ep, edges_men, entity_name_by_id)

except Exception as e:
    st.error(f"Failed to load/parse files: {e}")
    st.stop()

# Summary
total_eps = len(episode_df)
with_triples = int((episode_df["n_rel_edges"] > 0).sum())
st.subheader("Dataset summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Episodic chunks", total_eps)
c2.metric("Chunks with ≥1 RELATES_TO", with_triples)
c3.metric("Total RELATES_TO edges matched", int(episode_df["n_rel_edges"].sum()))
c4.metric("Entity name column", entity_name_col)

# Filter view
view_df = episode_df.copy()
if show_only_with_triples:
    view_df = view_df[view_df["n_rel_edges"] > 0].reset_index(drop=True)

# Table
st.subheader("Chunks (click a row by selecting an episodic_id below)")
table_cols = ["episodic_id", "n_rel_edges", "edge_uuids_total", "n_mentions", "sent"]
st.dataframe(view_df[table_cols], use_container_width=True, height=260)

# Selector
st.subheader("Explore a specific chunk")
selected_ep = st.selectbox(
    "episodic_id",
    options=view_df["episodic_id"].astype(int).tolist(),
    index=0 if len(view_df) else None,
)

row = episode_df[episode_df["episodic_id"] == selected_ep].iloc[0]

left, right = st.columns([1.2, 1.0])

with left:
    st.markdown("### Chunk text")
    st.write(row["sent"])

    st.markdown("### Triples (RELATES_TO edges linked by UUID)")
    triples = row["triples"]
    if not triples:
        st.warning("No RELATES_TO triples linked to this chunk (via entity_edges UUIDs).")
    else:
        triples_df = pd.DataFrame(triples, columns=["subject", "predicate", "object"])
        st.dataframe(triples_df, use_container_width=True, height=260)

with right:
    st.markdown("### Edge UUID diagnostics")
    st.write(f"entity_edges UUIDs in chunk: **{row['edge_uuids_total']}**")
    st.write(f"UUIDs matched in edges_RELATES_TO: **{row['n_rel_edges']}**")
    st.code("\n".join(row["rel_uuids"]) if row["rel_uuids"] else "(none)", language="text")

    st.markdown("### Mentions (optional)")
    if "mention_entities" in episode_df.columns:
        st.write(f"Unique mentioned entities: **{row['n_mentions']}**")
        st.code("\n".join(row["mention_entities"]) if row["mention_entities"] else "(none)", language="text")

# Expanded edges view
st.subheader("RELATES_TO edges expanded (for the selected chunk)")
if expanded_rel.empty:
    st.info("No RELATES_TO edges matched any episodic.entity_edges UUIDs.")
else:
    exp = expanded_rel[expanded_rel["episodic_id"] == selected_ep].copy()
    if exp.empty:
        st.info("No RELATES_TO edges for this chunk.")
    else:
        st.dataframe(exp, use_container_width=True, height=260)

# Export
st.subheader("Export")
colA, colB = st.columns(2)

with colA:
    jsonl_data = _jsonl_bytes(episode_df, id_style=id_style)
    st.download_button(
        "Download JSONL (id, sent, triples)",
        data=jsonl_data,
        file_name="episodic_triples.jsonl",
        mime="application/jsonl",
    )
    st.caption("This JSONL is directly compatible with your evaluator’s expected structure (id/sent/triples).")

with colB:
    csv_data = _csv_bytes(episode_df)
    st.download_button(
        "Download CSV (episodic_id, sent, triples, mentions, counts)",
        data=csv_data,
        file_name="episodic_triples.csv",
        mime="text/csv",
    )
    st.caption("Triples and mentions are JSON-encoded strings inside the CSV.")

st.divider()
st.subheader("Sanity checks")
st.write("These help verify you’re joining things correctly.")

# 1) How many UUIDs in episodic entity_edges are not found in RELATES_TO?
all_rel_uuids = set(edges_rel["uuid"].astype(str).tolist())
unknown_counts = []
for _, r in episode_df.iterrows():
    ep_uuids = set(r["rel_uuids"])  # already intersected
    # compute unknowns by re-parsing from original might be expensive; approximate:
    # unknowns = total - matched_rel - (maybe mentions) -- here we show matched vs total
    unknown_counts.append(int(r["edge_uuids_total"]) - int(r["n_rel_edges"]))
episode_df_sc = episode_df.copy()
episode_df_sc["non_rel_uuids_count_est"] = unknown_counts

st.dataframe(
    episode_df_sc[["episodic_id", "edge_uuids_total", "n_rel_edges", "non_rel_uuids_count_est", "n_mentions"]]
    .sort_values(["n_rel_edges", "edge_uuids_total"], ascending=False)
    .head(50),
    use_container_width=True,
    height=260,
)
st.caption(
    "non_rel_uuids_count_est = entity_edges UUID count - RELATES_TO UUID matches. "
    "If you uploaded MENTIONS, most of those remaining UUIDs are likely mention edges."
)
