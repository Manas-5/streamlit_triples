import io
import ast
import colorsys

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import umap
import hdbscan  # pip install hdbscan
from sklearn.preprocessing import normalize

# -------------------- Streamlit setup --------------------

st.set_page_config(page_title="Nodes & Edges Exporter", layout="wide")
st.title("Nodes & Edges Export Helper")

st.markdown(
    """
This app expects **two CSV files**:

- `nodes_Entity.csv` with at least: `id`, `uuid`, `name`, `summary`, `name_embedding`
- `edges_RELATES_TO_with_name_emb.csv` with at least:
  `id`, `uuid`, `from_id`, `to_id`, `name`, `fact`,
  `fact_embedding`, `edge_name_embedding`

It will generate:

- Downloadable **.csv** and **.txt** files for nodes & edges
- Edge uniqueness stats (UUID, structure, name distribution)
- UMAPs + HDBSCAN for:
  - **Node name embeddings** (`name_embedding`)
  - **Edge fact embeddings** (`fact_embedding`)
  - **Edge relation-name embeddings** (`edge_name_embedding`)
"""
)

# -------------------- Helper functions --------------------


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    DataFrame → CSV bytes (UTF-8 with BOM).
    Ensures accent characters display correctly in Excel.
    """
    csv_text = df.to_csv(index=False)
    return ("\ufeff" + csv_text).encode("utf-8-sig")


def df_to_txt_bytes(df: pd.DataFrame) -> bytes:
    """
    DataFrame → TXT bytes (UTF-8, tab-separated).
    Accents are safely preserved.
    """
    txt_text = df.to_csv(index=False, sep="\t")
    return txt_text.encode("utf-8")


def add_serial_numbers(df: pd.DataFrame, col_name: str = "S.No") -> pd.DataFrame:
    df = df.copy()
    df.insert(0, col_name, range(1, len(df) + 1))
    return df


def parse_embedding_column(df: pd.DataFrame, col_name: str):
    """
    Parse an embedding column from the CSV into a (N, D) numpy array.

    Supports:
    - Stringified Python lists: "[0.1, 0.2, ...]"
    - Space or comma-separated floats: "0.1 0.2 ..." or "0.1,0.2,..."
    - Actual Python lists/tuples/ndarrays (if already parsed)

    Returns:
        X: np.ndarray of shape (N, D)
        df_valid: DataFrame of rows that had valid embeddings (index reset)
    """
    if col_name not in df.columns:
        raise KeyError(f"Column '{col_name}' not found in DataFrame.")

    rows = []
    embs = []

    for idx, v in df[col_name].items():
        if pd.isna(v):
            continue

        try:
            if isinstance(v, (list, tuple, np.ndarray)):
                vec = np.array(v, dtype="float32")
            elif isinstance(v, str):
                s = v.strip()
                try:
                    # Try list-like string first
                    vec = np.array(ast.literal_eval(s), dtype="float32")
                except Exception:
                    # Fallback: split by whitespace / commas
                    parts = s.replace(",", " ").split()
                    vec = np.array([float(p) for p in parts], dtype="float32")
            else:
                continue
        except Exception:
            continue

        embs.append(vec)
        rows.append(idx)

    if not embs:
        raise ValueError(f"No valid embeddings parsed from column '{col_name}'.")

    X = np.vstack(embs)
    df_valid = df.loc[rows].reset_index(drop=True)
    return X, df_valid


def compute_umap(
    X: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    metric: str = "euclidean",
    random_state: int = 42,
):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(X)


def fig_to_png_bytes(fig) -> bytes:
    """Convert a Matplotlib figure to PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=250)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def rgb_to_hex(r: float, g: float, b: float) -> str:
    """
    Convert RGB floats in [0, 1] to hex string "#RRGGBB".
    """
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


def generate_hsv_hex_colors(n: int, saturation: float = 0.7, value: float = 1.0):
    """
    Generate n distinct colors around the HSV color wheel.
    Returns list of hex strings.
    """
    if n <= 0:
        return []
    colors = []
    for i in range(n):
        h = i / n
        r, g, b = colorsys.hsv_to_rgb(h, saturation, value)
        colors.append(rgb_to_hex(r, g, b))
    return colors


def generate_neon_hex_colors(n: int, seed: int = 0):
    """
    Generate n neon-like colors (bright, high contrast) as hex strings.
    """
    if n <= 0:
        return []
    rng = np.random.default_rng(seed)
    colors = []
    for _ in range(n):
        c = rng.random(3) ** 0.3  # gamma correction for neon feel
        colors.append(rgb_to_hex(c[0], c[1], c[2]))
    return colors


# -------------------- File uploaders --------------------

st.sidebar.header("Upload your files")

nodes_file = st.sidebar.file_uploader(
    "Upload nodes_Entity.csv",
    type=["csv"],
    key="nodes",
)

edges_file = st.sidebar.file_uploader(
    "Upload edges_RELATES_TO_with_name_emb.csv",
    type=["csv"],
    key="edges",
)

nodes_df = pd.read_csv(nodes_file) if nodes_file is not None else None
edges_df = pd.read_csv(edges_file) if edges_file is not None else None

# -------------------- Nodes section --------------------

st.header("Nodes (from nodes_Entity.csv)")

if nodes_df is None:
    st.info("Upload **nodes_Entity.csv** in the sidebar to see node downloads.")
else:
    st.subheader("Preview of nodes")
    st.dataframe(nodes_df.head())

    # --- basic node downloads ---

    nodes_names_df = add_serial_numbers(nodes_df[["name"]])

    if "summary" in nodes_df.columns:
        nodes_names_summ_df = add_serial_numbers(nodes_df[["name", "summary"]])
    else:
        nodes_names_summ_df = None
        st.warning(
            "Column 'summary' not found in nodes_Entity.csv, "
            "so I can't build the name+summary outputs."
        )

    st.subheader("Downloads for nodes")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Node names (with serial numbers)**")
        st.download_button(
            label="⬇️ Download node names (CSV)",
            data=df_to_csv_bytes(nodes_names_df),
            file_name="nodes_names.csv",
            mime="text/csv",
        )
    with col2:
        st.write(" ")
        st.download_button(
            label="⬇️ Download node names (TXT)",
            data=df_to_txt_bytes(nodes_names_df),
            file_name="nodes_names.txt",
            mime="text/plain",
        )

    if nodes_names_summ_df is not None:
        col3, col4 = st.columns(2)
        with col3:
            st.write("**Node names + summaries (with serial numbers)**")
            st.download_button(
                label="⬇️ Download node names & summaries (CSV)",
                data=df_to_csv_bytes(nodes_names_summ_df),
                file_name="nodes_names_summaries.csv",
                mime="text/csv",
            )
        with col4:
            st.write(" ")
            st.download_button(
                label="⬇️ Download node names & summaries (TXT)",
                data=df_to_txt_bytes(nodes_names_summ_df),
                file_name="nodes_names_summaries.txt",
                mime="text/plain",
            )

    # ---------- UMAP + HDBSCAN for node name embeddings ----------

    st.subheader("UMAP + HDBSCAN for node name embeddings (name_embedding)")

    if "name_embedding" not in nodes_df.columns:
        st.warning(
            "Column 'name_embedding' not found in nodes_Entity.csv, "
            "so I can't build UMAPs for node name embeddings."
        )
    else:
        with st.expander("UMAP + HDBSCAN parameters for nodes", expanded=False):
            # UMAP
            n_neighbors_nodes = st.slider(
                "UMAP n_neighbors (nodes)",
                min_value=5,
                max_value=100,
                value=15,
                step=1,
            )
            min_dist_nodes = st.slider(
                "UMAP min_dist (nodes)",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.01,
            )
            metric_nodes = st.selectbox(
                "UMAP distance metric (nodes)",
                options=["euclidean", "manhattan"],  # no cosine here
                index=0,
            )
            random_state_nodes = st.number_input(
                "Random seed (nodes UMAP)",
                value=42,
            )

            # HDBSCAN on UMAP space
            color_mode_nodes = st.radio(
                "Color mode for node UMAP",
                options=["None", "Cluster (HDBSCAN)"],
                index=0,
            )

            hdbscan_min_cluster_size_nodes = st.slider(
                "HDBSCAN min_cluster_size (nodes, on UMAP coords)",
                min_value=5,
                max_value=500,
                value=20,  # 40
                step=5,
            )
            hdbscan_min_samples_nodes = st.slider(
                "HDBSCAN min_samples (nodes, on UMAP coords)",
                min_value=1,
                max_value=200,
                value=5,   # 20
                step=1,
            )
            hdbscan_cluster_method_nodes = st.radio(
                "HDBSCAN cluster_selection_method (nodes)",
                options=["eom", "leaf"],
                index=0,
            )

            color_palette_nodes = st.radio(
                "Color palette for nodes",
                options=["Dark24", "HSV wheel", "Neon"],
                index=0,
            )

        if st.button("Compute UMAP for node name embeddings"):
            try:
                # Parse high-dimensional embeddings
                X_nodes, nodes_valid = parse_embedding_column(
                    nodes_df, "name_embedding"
                )

                # 2D UMAP (used both for visualization and clustering space)
                coords_2d_nodes = compute_umap(
                    X_nodes,
                    n_neighbors=n_neighbors_nodes,
                    min_dist=min_dist_nodes,
                    n_components=2,
                    metric=metric_nodes,
                    random_state=int(random_state_nodes),
                )
                nodes_valid_2d = nodes_valid.copy()
                nodes_valid_2d["UMAP_1"] = coords_2d_nodes[:, 0]
                nodes_valid_2d["UMAP_2"] = coords_2d_nodes[:, 1]

                node_labels = None
                legend_labels_nodes = []
                color_dict_nodes = {}
                color_source_nodes = None
                # clusterer_nodes = None  # added feature for HDBSCAN
                # HDBSCAN clustering in UMAP 2D space
                if color_mode_nodes == "Cluster (HDBSCAN)":
                    clusterer_nodes = hdbscan.HDBSCAN(
                        min_cluster_size=hdbscan_min_cluster_size_nodes,
                        min_samples=hdbscan_min_samples_nodes,
                        metric="euclidean",  # distance in UMAP space
                        cluster_selection_method=hdbscan_cluster_method_nodes,
                    )
                    node_labels = clusterer_nodes.fit_predict(coords_2d_nodes)

                    # If everything is noise (-1), collapse into a single cluster
                    if np.unique(node_labels).size == 1 and np.unique(node_labels)[0] == -1:
                        node_labels[:] = 0

                    nodes_valid_2d["cluster"] = node_labels.astype(str)
                    color_source_nodes = "cluster"
                    legend_labels_nodes = nodes_valid_2d["cluster"].unique()

                    if len(legend_labels_nodes) > 0:
                        if color_palette_nodes == "Dark24":
                            base_colors = px.colors.qualitative.Dark24
                            colors = [
                                base_colors[i % len(base_colors)]
                                for i in range(len(legend_labels_nodes))
                            ]
                        elif color_palette_nodes == "HSV wheel":
                            colors = generate_hsv_hex_colors(len(legend_labels_nodes))
                        else:  # Neon
                            colors = generate_neon_hex_colors(len(legend_labels_nodes))
                        color_dict_nodes = dict(zip(legend_labels_nodes, colors))

                # -------- 2D interactive (Plotly) --------
                st.markdown("### 2D UMAP (interactive)")

                if color_source_nodes is not None and len(color_dict_nodes) > 0:
                    color_col_nodes_2d = color_source_nodes
                    color_discrete_map_nodes_2d = color_dict_nodes
                else:
                    color_col_nodes_2d = None
                    color_discrete_map_nodes_2d = None

                fig_nodes_2d = px.scatter(
                    nodes_valid_2d,
                    x="UMAP_1",
                    y="UMAP_2",
                    color=color_col_nodes_2d,
                    color_discrete_map=color_discrete_map_nodes_2d,
                    hover_name="name",
                    title="Node name embeddings - 2D UMAP",
                    height=600,
                )
                fig_nodes_2d.update_layout(
                    plot_bgcolor="black",
                    paper_bgcolor="black",
                    font=dict(color="white"),
                )
                st.plotly_chart(fig_nodes_2d, use_container_width=True)

                # -------- 2D static PNG (Matplotlib) --------
                fig_static, ax = plt.subplots(figsize=(8, 8))

                if color_source_nodes is not None and len(color_dict_nodes) > 0:
                    point_colors_nodes = nodes_valid_2d[color_source_nodes].map(
                        color_dict_nodes
                    )
                else:
                    point_colors_nodes = "white"

                ax.scatter(
                    nodes_valid_2d["UMAP_1"],
                    nodes_valid_2d["UMAP_2"],
                    c=point_colors_nodes,
                    s=3,             # small points
                    alpha=0.4,       # transparent for density
                    rasterized=True,
                    edgecolors="none",
                )
                ax.set_facecolor("black")
                ax.set_xlabel("UMAP 1", color="white")
                ax.set_ylabel("UMAP 2", color="white")
                ax.set_title("Node name embeddings - 2D UMAP (static)", color="white")
                ax.tick_params(colors="white")
                ax.set_aspect("equal", "box")
                fig_static.tight_layout()

                if (
                    len(legend_labels_nodes) > 0
                    and len(legend_labels_nodes) <= 30
                    and len(color_dict_nodes) > 0
                ):
                    for label in legend_labels_nodes:
                        ax.scatter([], [], c=[color_dict_nodes[label]], label=str(label))
                    ax.legend(
                        title="Node clusters",
                        loc="upper right",
                        facecolor="black",
                        edgecolor="white",
                        labelcolor="white",
                        title_fontsize=10,
                        fontsize=8,
                    )

                png_bytes_nodes = fig_to_png_bytes(fig_static)

                st.image(
                    png_bytes_nodes,
                    caption="Node name embeddings - 2D UMAP (static, colored)",
                    use_container_width=True,
                )
                st.download_button(
                    label="⬇️ Download 2D UMAP (nodes) as PNG",
                    data=png_bytes_nodes,
                    file_name="nodes_name_embedding_umap_2d.png",
                    mime="image/png",
                )

                # -------- 3D UMAP (Plotly) --------
                st.markdown("### 3D UMAP (interactive)")
                coords_3d_nodes = compute_umap(
                    X_nodes,
                    n_neighbors=n_neighbors_nodes,
                    min_dist=min_dist_nodes,
                    n_components=3,
                    metric=metric_nodes,
                    random_state=int(random_state_nodes),
                )
                nodes_valid_3d = nodes_valid.copy()
                nodes_valid_3d["UMAP_1"] = coords_3d_nodes[:, 0]
                nodes_valid_3d["UMAP_2"] = coords_3d_nodes[:, 1]
                nodes_valid_3d["UMAP_3"] = coords_3d_nodes[:, 2]

                if color_source_nodes == "cluster" and node_labels is not None:
                    nodes_valid_3d["cluster"] = node_labels.astype(str)
                    color_col_nodes_3d = "cluster"
                    color_discrete_map_nodes_3d = color_dict_nodes
                else:
                    color_col_nodes_3d = None
                    color_discrete_map_nodes_3d = None

                fig_nodes_3d = px.scatter_3d(
                    nodes_valid_3d,
                    x="UMAP_1",
                    y="UMAP_2",
                    z="UMAP_3",
                    color=color_col_nodes_3d,
                    color_discrete_map=color_discrete_map_nodes_3d,
                    hover_name="name",
                    title="Node name embeddings - 3D UMAP",
                    height=700,
                )
                fig_nodes_3d.update_layout(
                    scene=dict(
                        xaxis_backgroundcolor="black",
                        yaxis_backgroundcolor="black",
                        zaxis_backgroundcolor="black",
                    ),
                    paper_bgcolor="black",
                    font=dict(color="white"),
                )
                st.plotly_chart(fig_nodes_3d, use_container_width=True)

            except Exception as e:
                st.error(f"Error computing node UMAP: {e}")


# -------------------- Edges section --------------------

st.header("Edges (from edges_RELATES_TO_with_name_emb.csv)")

if edges_df is None:
    st.info("Upload **edges_RELATES_TO_with_name_emb.csv** in the sidebar to see edge downloads.")
else:
    st.subheader("Preview of edges")
    st.dataframe(edges_df.head())

    # ---------- Edge uniqueness / statistics ----------

    st.subheader("Edge uniqueness overview")

    if "uuid" in edges_df.columns:
        total_edges = len(edges_df)
        unique_uuids = edges_df["uuid"].nunique()
        duplicate_uuid_count = total_edges - unique_uuids

        if "name" in edges_df.columns:
            unique_uuid_name_count = (
                edges_df[["uuid", "name"]].drop_duplicates().shape[0]
            )
        else:
            unique_uuid_name_count = None

        colu1, colu2, colu3, colu4 = st.columns(4)
        with colu1:
            st.metric("Total edges (rows)", total_edges)
        with colu2:
            st.metric("Unique edges (UUID)", unique_uuids)
        with colu3:
            st.metric("Duplicate UUIDs", duplicate_uuid_count)
        with colu4:
            st.metric(
                "Unique (UUID, name) pairs",
                unique_uuid_name_count if unique_uuid_name_count is not None else "n/a",
            )

        with st.expander("More edge uniqueness stats & name distribution", expanded=False):
            if "name" in edges_df.columns:
                st.write(f"**Unique edge names**: {edges_df['name'].nunique()}")

                name_counts = (
                    edges_df["name"]
                    .value_counts()
                    .reset_index(name="count")
                )
                name_counts.columns = ["name", "count"]

                st.write("### Edge name frequency (top N)")
                top_k = st.slider(
                    "Number of top edge names to show in histogram",
                    min_value=5,
                    max_value=min(100, len(name_counts)),
                    value=min(30, len(name_counts)),
                    step=1,
                )
                fig_hist = px.bar(
                    name_counts.head(top_k),
                    x="name",
                    y="count",
                    title="Top edge names by frequency",
                )
                fig_hist.update_layout(xaxis_title="Edge name", yaxis_title="Count")
                st.plotly_chart(fig_hist, use_container_width=True)

                st.download_button(
                    label="⬇️ Download edge name counts (CSV)",
                    data=df_to_csv_bytes(name_counts),
                    file_name="edge_name_counts.csv",
                    mime="text/csv",
                )

            if "fact" in edges_df.columns:
                st.write(f"**Unique facts**: {edges_df['fact'].nunique()}")

            if {"from_id", "to_id"}.issubset(edges_df.columns):
                st.write(
                    f"**Unique (from_id, to_id) pairs**: "
                    f"{edges_df[['from_id', 'to_id']].drop_duplicates().shape[0]}"
                )
            if {"from_id", "to_id", "name"}.issubset(edges_df.columns):
                st.write(
                    f"**Unique (from_id, to_id, name) triples**: "
                    f"{edges_df[['from_id', 'to_id', 'name']].drop_duplicates().shape[0]}"
                )

            if {"uuid", "name"}.issubset(edges_df.columns):
                edges_unique_by_uuid_name = edges_df.drop_duplicates(
                    subset=["uuid", "name"]
                )
                edges_unique_by_uuid_name_serial = add_serial_numbers(
                    edges_unique_by_uuid_name
                )

                st.write("### Unique edges by (UUID, name)")
                st.dataframe(edges_unique_by_uuid_name_serial.head(50))

                st.download_button(
                    label="⬇️ Download unique (UUID, name) edges (CSV)",
                    data=df_to_csv_bytes(edges_unique_by_uuid_name_serial),
                    file_name="edges_unique_by_uuid_and_name.csv",
                    mime="text/csv",
                )

            if duplicate_uuid_count > 0:
                st.write("### Duplicate UUID rows")
                duplicate_edges = edges_df[edges_df.duplicated("uuid", keep=False)].copy()
                duplicate_edges = duplicate_edges.sort_values("uuid")
                duplicate_edges_serial = add_serial_numbers(duplicate_edges)
                st.dataframe(duplicate_edges_serial.head(50))

                st.download_button(
                    label="⬇️ Download all rows with duplicate UUIDs (CSV)",
                    data=df_to_csv_bytes(duplicate_edges_serial),
                    file_name="edges_duplicate_uuid_rows.csv",
                    mime="text/csv",
                )
            else:
                st.success("No duplicate UUIDs found in edges.")
    else:
        st.warning(
            "Column 'uuid' not found in edges_RELATES_TO_with_name_emb.csv, "
            "can't compute UUID-based uniqueness."
        )

    # ---------- Basic edge exports ----------

    edges_names_df = add_serial_numbers(edges_df[["name"]])

    if "fact" in edges_df.columns:
        edges_name_fact_df = add_serial_numbers(edges_df[["name", "fact"]])
    else:
        edges_name_fact_df = None
        st.warning(
            "Column 'fact' not found in edges_RELATES_TO_with_name_emb.csv, "
            "so I can't build the name+fact outputs."
        )

    st.subheader("Downloads for edges (basic)")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Edge names (with serial numbers)**")
        st.download_button(
            label="⬇️ Download edge names (CSV)",
            data=df_to_csv_bytes(edges_names_df),
            file_name="edges_names.csv",
            mime="text/csv",
        )
    with col2:
        st.write(" ")
        st.download_button(
            label="⬇️ Download edge names (TXT)",
            data=df_to_txt_bytes(edges_names_df),
            file_name="edges_names.txt",
            mime="text/plain",
        )

    if edges_name_fact_df is not None:
        col3, col4 = st.columns(2)
        with col3:
            st.write("**Edge names + facts (with serial numbers)**")
            st.download_button(
                label="⬇️ Download edge names & facts (CSV)",
                data=df_to_csv_bytes(edges_name_fact_df),
                file_name="edges_names_facts.csv",
                mime="text/csv",
            )
        with col4:
            st.write(" ")
            st.download_button(
                label="⬇️ Download edge names & facts (TXT)",
                data=df_to_txt_bytes(edges_name_fact_df),
                file_name="edges_names_facts.txt",
                mime="text/plain",
            )

    # ---------- UMAP for edge fact embeddings (galaxy + HDBSCAN) ----------

    st.subheader("UMAP + HDBSCAN for edge fact embeddings (fact_embedding)")

    if "fact_embedding" not in edges_df.columns:
        st.warning(
            "Column 'fact_embedding' not found in edges_RELATES_TO_with_name_emb.csv, "
            "so I can't build UMAPs for edge fact embeddings."
        )
    else:
        with st.expander(
            "UMAP + HDBSCAN parameters for edge facts (galaxy view)",
            expanded=False,
        ):
            n_neighbors_edges = st.slider(
                "UMAP n_neighbors (edges, facts)",
                min_value=5,
                max_value=100,
                value=30,
                step=1,
            )
            min_dist_edges = st.slider(
                "UMAP min_dist (edges, facts)",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.01,
            )
            metric_edges = st.selectbox(
                "UMAP distance metric (edges, facts)",
                options=["euclidean", "manhattan"],
                index=0,
            )
            random_state_edges = st.number_input(
                "Random seed (edges UMAP, facts)",
                value=42,
            )

            hdbscan_min_cluster_size = st.slider(
                "HDBSCAN min_cluster_size (edges, on UMAP coords)",
                min_value=5,
                max_value=500,
                value=30,  # 60
                step=5,
            )
            hdbscan_min_samples = st.slider(
                "HDBSCAN min_samples (edges, on UMAP coords)",
                min_value=1,
                max_value=200,
                value=5,  # 30
                step=1,
            )
            hdbscan_cluster_selection_method = st.radio(
                "HDBSCAN cluster_selection_method (edges, facts)",
                options=["eom", "leaf"],
                index=0,
            )

            color_mode_edges = st.radio(
                "Color mode for 2D UMAP (edges, facts)",
                options=["Edge name", "Cluster (HDBSCAN)", "None"],
                index=1,
            )

            color_palette_edges = st.radio(
                "Color palette for edges (facts)",
                options=["Dark24", "HSV wheel", "Neon"],
                index=0,
            )

            max_points = st.slider(
                "Max points to display in interactive 2D UMAP (facts)",
                min_value=500,
                max_value=10000,
                value=3000,
                step=500,
            )

            point_size = st.slider(
                "Point size (2D/3D UMAP, facts)",
                min_value=2,
                max_value=10,
                value=4,
                step=1,
            )

            point_opacity = st.slider(
                "Point opacity (2D/3D UMAP, facts)",
                min_value=0.1,
                max_value=1.0,
                value=0.7,
                step=0.05,
            )

        if st.button("Compute UMAP for edge fact embeddings"):
            try:
                # High-D embeddings for facts
                X_edges, edges_valid = parse_embedding_column(
                    edges_df, "fact_embedding"
                )

                # 2D UMAP on full set (used for clustering and visualization)
                coords_2d_edges = compute_umap(
                    X_edges,
                    n_neighbors=n_neighbors_edges,
                    min_dist=min_dist_edges,
                    n_components=2,
                    metric=metric_edges,
                    random_state=int(random_state_edges),
                )
                edges_valid_2d = edges_valid.copy()
                edges_valid_2d["UMAP_1"] = coords_2d_edges[:, 0]
                edges_valid_2d["UMAP_2"] = coords_2d_edges[:, 1]

                # Clustering / color choice
                cluster_labels = None
                color_source_col = None
                legend_labels = []
                color_dict_edges = {}
                # clusterer_edges = None # added feature for
                if color_mode_edges == "Cluster (HDBSCAN)":
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=hdbscan_min_cluster_size,
                        min_samples=hdbscan_min_samples,
                        metric="euclidean",  # on UMAP 2D coords
                        cluster_selection_method=hdbscan_cluster_selection_method,
                    )
                    cluster_labels = clusterer.fit_predict(coords_2d_edges)
                    if np.unique(cluster_labels).size == 1 and np.unique(cluster_labels)[0] == -1:
                        cluster_labels[:] = 0
                    edges_valid_2d["cluster"] = cluster_labels.astype(str)
                    color_source_col = "cluster"
                    legend_labels = edges_valid_2d["cluster"].unique()

                elif color_mode_edges == "Edge name" and "name" in edges_valid_2d.columns:
                    color_source_col = "name"
                    legend_labels = edges_valid_2d["name"].unique()
                else:
                    color_source_col = None
                    legend_labels = []

                if color_source_col is not None and len(legend_labels) > 0:
                    if color_palette_edges == "Dark24":
                        base_colors = px.colors.qualitative.Dark24
                        colors = [
                            base_colors[i % len(base_colors)]
                            for i in range(len(legend_labels))
                        ]
                    elif color_palette_edges == "HSV wheel":
                        colors = generate_hsv_hex_colors(len(legend_labels))
                    else:
                        colors = generate_neon_hex_colors(len(legend_labels))
                    color_dict_edges = dict(zip(legend_labels, colors))

                # Subsample for interactive plot
                if len(edges_valid_2d) > max_points:
                    edges_valid_2d_sample = edges_valid_2d.sample(
                        max_points, random_state=int(random_state_edges)
                    )
                else:
                    edges_valid_2d_sample = edges_valid_2d

                # -------- 2D interactive (Plotly) --------
                st.markdown("### 2D UMAP (interactive) – fact galaxy")

                if color_source_col is not None and len(color_dict_edges) > 0:
                    color_col_edges_2d = color_source_col
                    color_discrete_map_edges_2d = color_dict_edges
                else:
                    color_col_edges_2d = None
                    color_discrete_map_edges_2d = None

                fig_edges_2d = px.scatter(
                    edges_valid_2d_sample,
                    x="UMAP_1",
                    y="UMAP_2",
                    color=color_col_edges_2d,
                    color_discrete_map=color_discrete_map_edges_2d,
                    hover_name="name" if "name" in edges_valid_2d_sample.columns else None,
                    hover_data=["fact"] if "fact" in edges_valid_2d_sample.columns else None,
                    title="Edge fact embeddings - 2D UMAP",
                    height=650,
                    render_mode="webgl",
                )
                fig_edges_2d.update_traces(
                    marker=dict(size=point_size, opacity=point_opacity)
                )
                fig_edges_2d.update_layout(
                    legend_title_text=(
                        "Edge name"
                        if color_mode_edges == "Edge name"
                        else "Cluster (HDBSCAN)"
                        if color_mode_edges == "Cluster (HDBSCAN)"
                        else ""
                    ),
                    plot_bgcolor="black",
                    paper_bgcolor="black",
                    font=dict(color="white"),
                )
                st.plotly_chart(fig_edges_2d, use_container_width=True)

                # -------- 2D static PNG (Matplotlib) --------
                fig_static_e, ax_e = plt.subplots(figsize=(9, 9))

                if color_source_col is not None and len(color_dict_edges) > 0:
                    point_colors_e = edges_valid_2d[color_source_col].map(
                        color_dict_edges
                    )
                else:
                    point_colors_e = "white"

                ax_e.scatter(
                    edges_valid_2d["UMAP_1"],
                    edges_valid_2d["UMAP_2"],
                    c=point_colors_e,
                    s=3,
                    alpha=0.4,
                    rasterized=True,
                    edgecolors="none",
                )

                ax_e.set_facecolor("black")
                ax_e.set_xlabel("UMAP 1", color="white")
                ax_e.set_ylabel("UMAP 2", color="white")
                ax_e.set_title("Edge fact embeddings - 2D UMAP (static)", color="white")
                ax_e.tick_params(colors="white")
                ax_e.set_aspect("equal", "box")
                fig_static_e.tight_layout()

                if (
                    len(legend_labels) > 0
                    and len(legend_labels) <= 30
                    and len(color_dict_edges) > 0
                ):
                    for label in legend_labels:
                        ax_e.scatter([], [], c=[color_dict_edges[label]], label=str(label))
                    ax_e.legend(
                        title="Legend",
                        loc="upper right",
                        facecolor="black",
                        edgecolor="white",
                        labelcolor="white",
                        title_fontsize=10,
                        fontsize=8,
                    )

                png_bytes_edges_scatter = fig_to_png_bytes(fig_static_e)

                st.image(
                    png_bytes_edges_scatter,
                    caption="Edge fact embeddings - 2D UMAP (static, colored)",
                    use_container_width=True,
                )

                st.download_button(
                    label="⬇️ Download 2D UMAP (edges, facts) – colored scatter PNG",
                    data=png_bytes_edges_scatter,
                    file_name="edges_fact_embedding_umap_2d_colored.png",
                    mime="image/png",
                )

                # -------- Distribution PNG (same colors) --------
                if (
                    len(legend_labels) > 0
                    and color_source_col is not None
                    and len(color_dict_edges) > 0
                ):
                    counts = (
                        edges_valid_2d[color_source_col]
                        .value_counts()
                        .reindex(legend_labels)
                    )

                    def shorten(lbl, maxlen=20):
                        s = str(lbl)
                        return s if len(s) <= maxlen else s[: maxlen - 3] + "..."

                    short_labels = [shorten(l) for l in legend_labels]

                    fig_bar, ax_bar = plt.subplots(
                        figsize=(max(6, len(legend_labels) * 0.35), 4)
                    )
                    bar_colors = [color_dict_edges[l] for l in legend_labels]

                    ax_bar.bar(range(len(legend_labels)), counts.values, color=bar_colors)
                    ax_bar.set_xticks(range(len(legend_labels)))
                    ax_bar.set_xticklabels(short_labels, rotation=90, fontsize=8)
                    ax_bar.set_ylabel("Count")
                    title_text = (
                        "Edge relation distribution by name"
                        if color_source_col == "name"
                        else "Edge relation distribution by HDBSCAN cluster"
                    )
                    ax_bar.set_title(title_text)

                    fig_bar.tight_layout()
                    png_bytes_edges_bar = fig_to_png_bytes(fig_bar)

                    st.image(
                        png_bytes_edges_bar,
                        caption="Relation / cluster distribution (matching colors)",
                        use_container_width=True,
                    )
                    st.download_button(
                        label="⬇️ Download relation distribution PNG",
                        data=png_bytes_edges_bar,
                        file_name="edges_relation_distribution.png",
                        mime="image/png",
                    )

                # -------- 3D UMAP (Plotly) --------
                st.markdown("### 3D UMAP (interactive) – fact galaxy")

                coords_3d_edges = compute_umap(
                    X_edges,
                    n_neighbors=n_neighbors_edges,
                    min_dist=min_dist_edges,
                    n_components=3,
                    metric=metric_edges,
                    random_state=int(random_state_edges),
                )
                edges_valid_3d = edges_valid.copy()
                edges_valid_3d["UMAP_1"] = coords_3d_edges[:, 0]
                edges_valid_3d["UMAP_2"] = coords_3d_edges[:, 1]
                edges_valid_3d["UMAP_3"] = coords_3d_edges[:, 2]

                # Assign labels into 3D df, if we clustered
                if color_source_col == "name" and "name" in edges_valid_3d.columns:
                    color_col_edges_3d = "name"
                elif color_source_col == "cluster" and cluster_labels is not None:
                    edges_valid_3d["cluster"] = cluster_labels.astype(str)
                    color_col_edges_3d = "cluster"
                else:
                    color_col_edges_3d = None

                if color_col_edges_3d is not None and len(color_dict_edges) > 0:
                    color_discrete_map_edges_3d = color_dict_edges
                else:
                    color_discrete_map_edges_3d = None

                fig_edges_3d = px.scatter_3d(
                    edges_valid_3d,
                    x="UMAP_1",
                    y="UMAP_2",
                    z="UMAP_3",
                    color=color_col_edges_3d,
                    color_discrete_map=color_discrete_map_edges_3d,
                    hover_name="name" if "name" in edges_valid_3d.columns else None,
                    hover_data=["fact"] if "fact" in edges_valid_3d.columns else None,
                    title="Edge fact embeddings - 3D UMAP",
                    height=750,
                )
                fig_edges_3d.update_traces(
                    marker=dict(size=point_size, opacity=point_opacity)
                )
                fig_edges_3d.update_layout(
                    legend_title_text=(
                        "Edge name"
                        if color_mode_edges == "Edge name"
                        else "Cluster (HDBSCAN)"
                        if color_mode_edges == "Cluster (HDBSCAN)"
                        else ""
                    ),
                    scene=dict(
                        xaxis_backgroundcolor="black",
                        yaxis_backgroundcolor="black",
                        zaxis_backgroundcolor="black",
                    ),
                    paper_bgcolor="black",
                    font=dict(color="white"),
                )
                st.plotly_chart(fig_edges_3d, use_container_width=True)

            except Exception as e:
                st.error(f"Error computing edge fact UMAP: {e}")

    # ---------- UMAP + HDBSCAN for edge relation-name embeddings (edge_name_embedding) ----------

    st.subheader("UMAP + HDBSCAN for edge *relation names* (edge_name_embedding)")

    if "edge_name_embedding" not in edges_df.columns:
        st.warning(
            "Column 'edge_name_embedding' not found in edges_RELATES_TO_with_name_emb.csv. "
            "Run your embedding script to add it."
        )
    else:
        with st.expander(
            "UMAP + HDBSCAN parameters for relation names",
            expanded=False,
        ):
            # UMAP params
            n_neighbors_rel = st.slider(
                "UMAP n_neighbors (relation names)",
                min_value=5,
                max_value=100,
                value=30,   # 30
                step=1,
            )
            min_dist_rel = st.slider(
                "UMAP min_dist (relation names)",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.01,
            )
            metric_rel = st.selectbox(
                "UMAP / HDBSCAN metric (relation names)",
                options=["euclidean", "manhattan"],
                index=0,
            )
            random_state_rel = st.number_input(
                "Random seed (relation names UMAP)",
                value=42,
            )

            # HDBSCAN params (on UMAP 2D coords of unique relation-name embeddings)
            min_cluster_size_rel = st.slider(
                "HDBSCAN min_cluster_size (relation names, on UMAP coords)",
                min_value=3,
                max_value=200,
                value=20,
                step=1,
            )
            min_samples_rel = st.slider(
                "HDBSCAN min_samples (relation names, on UMAP coords)",
                min_value=1,
                max_value=50,
                value=5,
                step=1,
            )
            cluster_method_rel = st.radio(
                "HDBSCAN cluster_selection_method (relation names)",
                options=["eom", "leaf"],
                index=1,
            )

            # Plot options
            dedupe_rel = st.checkbox(
                "Deduplicate relation names (one point per unique name)",
                value=False,
                help=(
                    "If checked, each relation name is shown once (size = how often it appears). "
                    "If unchecked, every edge is a point, but all edges of the same name "
                    "share the same cluster, color, and UMAP coordinates."
                ),
            )
            color_palette_rel = st.radio(
                "Color palette for relation-name clusters",
                options=["Dark24", "HSV wheel", "Neon"],
                index=0,
            )

        if st.button("Compute UMAP + HDBSCAN for relation names"):
            try:
                # ---- Base data: non-null name and embedding ----
                base_rel = edges_df.dropna(subset=["edge_name_embedding", "name"]).copy()

                # ---- Unique relation names: one embedding per name ----
                unique_rel = (
                    base_rel
                    .groupby("name", as_index=False)
                    .agg(
                        {
                            "edge_name_embedding": "first",
                            "fact": "count",  # how many times this relation appears
                        }
                    )
                )
                unique_rel.rename(columns={"fact": "count"}, inplace=True)

                # Parse embeddings for unique relation names
                X_rel, unique_valid = parse_embedding_column(
                    unique_rel, "edge_name_embedding"
                )
                # Keep name & count aligned with parsed rows
                unique_valid["name"] = unique_rel.loc[unique_valid.index, "name"].values
                unique_valid["count"] = unique_rel.loc[unique_valid.index, "count"].values

                # ---- UMAP on unique relation-name embeddings (2D) ----
                coords_2d_rel = compute_umap(
                    X_rel,
                    n_neighbors=n_neighbors_rel,
                    min_dist=min_dist_rel,
                    n_components=2,
                    metric=metric_rel,
                    random_state=int(random_state_rel),
                )
                unique_valid["UMAP_1"] = coords_2d_rel[:, 0]
                unique_valid["UMAP_2"] = coords_2d_rel[:, 1]

                # ---- HDBSCAN on UMAP 2D coords ----
                clusterer_rel = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size_rel,
                    min_samples=min_samples_rel,
                    metric="euclidean",  # in UMAP 2D space
                    cluster_selection_method=cluster_method_rel,
                )
                rel_labels = clusterer_rel.fit_predict(coords_2d_rel)
                unique_valid["cluster"] = rel_labels.astype(str)

                # Map: name -> (cluster, count, UMAP coords)
                name_to_cluster = dict(
                    zip(unique_valid["name"], unique_valid["cluster"])
                )
                name_to_count = dict(
                    zip(unique_valid["name"], unique_valid["count"])
                )
                name_to_x = dict(zip(unique_valid["name"], unique_valid["UMAP_1"]))
                name_to_y = dict(zip(unique_valid["name"], unique_valid["UMAP_2"]))

                # ---- Color mapping: one color per cluster ----
                legend_labels_rel = unique_valid["cluster"].unique()
                if len(legend_labels_rel) > 0:
                    if color_palette_rel == "Dark24":
                        base_colors = px.colors.qualitative.Dark24
                        colors = [
                            base_colors[i % len(base_colors)]
                            for i in range(len(legend_labels_rel))
                        ]
                    elif color_palette_rel == "HSV wheel":
                        colors = generate_hsv_hex_colors(len(legend_labels_rel))
                    else:
                        colors = generate_neon_hex_colors(len(legend_labels_rel))
                    color_dict_rel = dict(zip(legend_labels_rel, colors))
                else:
                    color_dict_rel = {}

                # ---- Build plot DataFrames ----
                if dedupe_rel:
                    # One point per relation name
                    df_plot_2d = unique_valid.copy()
                else:
                    # Every edge is a point, but we reuse cluster & coords of its name
                    df_plot_2d = base_rel.copy()
                    df_plot_2d["cluster"] = df_plot_2d["name"].map(name_to_cluster)
                    df_plot_2d["count"] = df_plot_2d["name"].map(name_to_count)
                    df_plot_2d["UMAP_1"] = df_plot_2d["name"].map(name_to_x)
                    df_plot_2d["UMAP_2"] = df_plot_2d["name"].map(name_to_y)

                # -------- 2D interactive (Plotly) --------
                st.markdown("### 2D UMAP (interactive) – relation-name clusters")

                fig_rel_2d = px.scatter(
                    df_plot_2d,
                    x="UMAP_1",
                    y="UMAP_2",
                    color="cluster",
                    color_discrete_map=color_dict_rel if len(color_dict_rel) > 0 else None,
                    size="count" if "count" in df_plot_2d.columns else None,
                    hover_name="name",
                    hover_data=["count"] if "count" in df_plot_2d.columns else None,
                    title="Relation-name embeddings - 2D UMAP (clustered with HDBSCAN on UMAP space)",
                    height=650,
                )
                fig_rel_2d.update_layout(
                    legend_title_text="HDBSCAN cluster",
                    plot_bgcolor="black",
                    paper_bgcolor="black",
                    font=dict(color="white"),
                )
                st.plotly_chart(fig_rel_2d, use_container_width=True)

                # -------- 2D static PNG (Matplotlib) --------
                fig_rel_static, ax_rel = plt.subplots(figsize=(8, 8))
                if len(color_dict_rel) > 0:
                    point_colors_rel = df_plot_2d["cluster"].map(color_dict_rel)
                else:
                    point_colors_rel = "white"

                ax_rel.scatter(
                    df_plot_2d["UMAP_1"],
                    df_plot_2d["UMAP_2"],
                    c=point_colors_rel,
                    s=3,
                    alpha=0.4,
                    rasterized=True,
                    edgecolors="none",
                )
                ax_rel.set_facecolor("black")
                ax_rel.set_xlabel("UMAP 1", color="white")
                ax_rel.set_ylabel("UMAP 2", color="white")
                title_txt = (
                    "Relation-name embeddings - 2D UMAP (unique names, HDBSCAN clusters on UMAP)"
                    if dedupe_rel
                    else "Relation-name embeddings - 2D UMAP (all edges, HDBSCAN clusters on UMAP)"
                )
                ax_rel.set_title(title_txt, color="white")
                ax_rel.tick_params(colors="white")
                ax_rel.set_aspect("equal", "box")
                fig_rel_static.tight_layout()

                if (
                    len(legend_labels_rel) > 0
                    and len(legend_labels_rel) <= 30
                    and len(color_dict_rel) > 0
                ):
                    for label in legend_labels_rel:
                        ax_rel.scatter([], [], c=[color_dict_rel[label]], label=str(label))
                    ax_rel.legend(
                        title="HDBSCAN cluster",
                        loc="upper right",
                        facecolor="black",
                        edgecolor="white",
                        labelcolor="white",
                        title_fontsize=10,
                        fontsize=8,
                    )

                png_bytes_rel = fig_to_png_bytes(fig_rel_static)

                st.image(
                    png_bytes_rel,
                    caption="Relation-name embeddings - 2D UMAP (static, HDBSCAN clusters on UMAP)",
                    use_container_width=True,
                )
                st.download_button(
                    label="⬇️ Download 2D UMAP (relation names) as PNG",
                    data=png_bytes_rel,
                    file_name="edges_relation_name_embedding_umap_2d.png",
                    mime="image/png",
                )

                # -------- 3D UMAP (Plotly) --------
                st.markdown("### 3D UMAP (interactive) – relation-name clusters")

                coords_3d_rel = compute_umap(
                    X_rel,
                    n_neighbors=n_neighbors_rel,
                    min_dist=min_dist_rel,
                    n_components=3,
                    metric=metric_rel,
                    random_state=int(random_state_rel),
                )
                unique_valid_3d = unique_valid.copy()
                unique_valid_3d["UMAP_1"] = coords_3d_rel[:, 0]
                unique_valid_3d["UMAP_2"] = coords_3d_rel[:, 1]
                unique_valid_3d["UMAP_3"] = coords_3d_rel[:, 2]

                # For 3D, we’ll just show unique relation names (cleaner)
                fig_rel_3d = px.scatter_3d(
                    unique_valid_3d,
                    x="UMAP_1",
                    y="UMAP_2",
                    z="UMAP_3",
                    color="cluster",
                    color_discrete_map=color_dict_rel if len(color_dict_rel) > 0 else None,
                    size="count",
                    hover_name="name",
                    hover_data=["count"],
                    title="Relation-name embeddings - 3D UMAP (HDBSCAN clusters from UMAP 2D space)",
                    height=750,
                )
                fig_rel_3d.update_traces(marker=dict(opacity=0.7))
                fig_rel_3d.update_layout(
                    scene=dict(
                        xaxis_backgroundcolor="black",
                        yaxis_backgroundcolor="black",
                        zaxis_backgroundcolor="black",
                    ),
                    paper_bgcolor="black",
                    font=dict(color="white"),
                )
                st.plotly_chart(fig_rel_3d, use_container_width=True)

            except Exception as e:
                st.error(f"Error computing relation-name UMAP: {e}")

    # ---------- Joined edges + node names ----------

    st.header("Edges joined with node names")

    if nodes_df is None:
        st.warning(
            "To map from_id / to_id to node names, upload **nodes_Entity.csv** as well."
        )
    else:
        id_to_name = nodes_df.set_index("id")["name"]
        edges_joined = edges_df.copy()
        edges_joined["from_name"] = edges_joined["from_id"].map(id_to_name)
        edges_joined["to_name"] = edges_joined["to_id"].map(id_to_name)

        edges_fromname_edgename_toname_df = pd.DataFrame(
            {
                "from_name": edges_joined["from_name"],
                "edge_name": edges_joined["name"],
                "to_name": edges_joined["to_name"],
            }
        )
        edges_fromname_edgename_toname_df = add_serial_numbers(
            edges_fromname_edgename_toname_df
        )

        if "fact" in edges_joined.columns:
            edges_fromname_fact_toname_df = pd.DataFrame(
                {
                    "from_name": edges_joined["from_name"],
                    "fact": edges_joined["fact"],
                    "to_name": edges_joined["to_name"],
                }
            )
            edges_fromname_fact_toname_df = add_serial_numbers(
                edges_fromname_fact_toname_df
            )
        else:
            edges_fromname_fact_toname_df = None
            st.warning(
                "Column 'fact' not found in edges_RELATES_TO_with_name_emb.csv, "
                "so I can't build from_name / fact / to_name outputs."
            )

        if "fact" in edges_joined.columns:
            edges_full_df = pd.DataFrame(
                {
                    "from_name": edges_joined["from_name"],
                    "edge_name": edges_joined["name"],
                    "to_name": edges_joined["to_name"],
                    "fact": edges_joined["fact"],
                }
            )
            edges_full_df = add_serial_numbers(edges_full_df)
        else:
            edges_full_df = None
            st.warning(
                "Column 'fact' not found in edges_RELATES_TO_with_name_emb.csv, "
                "so I can't build from_name / edge_name / to_name / fact outputs."
            )

        st.subheader("Downloads for edges joined with node names")

        col5, col6 = st.columns(2)
        with col5:
            st.write("**From node name – Edge name – To node name**")
            st.download_button(
                label="⬇️ Download from_name / edge_name / to_name (CSV)",
                data=df_to_csv_bytes(edges_fromname_edgename_toname_df),
                file_name="edges_fromname_edgename_toname.csv",
                mime="text/csv",
            )
        with col6:
            st.write(" ")
            st.download_button(
                label="⬇️ Download from_name / edge_name / to_name (TXT)",
                data=df_to_txt_bytes(edges_fromname_edgename_toname_df),
                file_name="edges_fromname_edgename_toname.txt",
                mime="text/plain",
            )

        if edges_fromname_fact_toname_df is not None:
            col7, col8 = st.columns(2)
            with col7:
                st.write("**From node name – Fact – To node name**")
                st.download_button(
                    label="⬇️ Download from_name / fact / to_name (CSV)",
                    data=df_to_csv_bytes(edges_fromname_fact_toname_df),
                    file_name="edges_fromname_fact_toname.csv",
                    mime="text/csv",
                )
            with col8:
                st.write(" ")
                st.download_button(
                    label="⬇️ Download from_name / fact / to_name (TXT)",
                    data=df_to_txt_bytes(edges_fromname_fact_toname_df),
                    file_name="edges_fromname_fact_toname.txt",
                    mime="text/plain",
                )

        if edges_full_df is not None:
            st.markdown("**From node name – Edge name – To node name – Fact**")
            col9, col10 = st.columns(2)
            with col9:
                st.download_button(
                    label="⬇️ Download from_name / edge_name / to_name / fact (CSV)",
                    data=df_to_csv_bytes(edges_full_df),
                    file_name="edges_fromname_edgename_toname_fact.csv",
                    mime="text/csv",
                )
            with col10:
                st.download_button(
                    label="⬇️ Download from_name / edge_name / to_name / fact (TXT)",
                    data=df_to_txt_bytes(edges_full_df),
                    file_name="edges_fromname_edgename_toname_fact.txt",
                    mime="text/plain",
                )
