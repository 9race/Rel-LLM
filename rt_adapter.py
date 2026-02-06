"""
Adapter module to integrate relational-transformer repository with Rel-LLM.
Reuses existing RT code without rewriting.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from functools import partial
import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType
import torch_frame
from torch_frame import stype
import numpy as np
import json

# Add relational-transformer to path
rt_path = Path(__file__).parent / "relational-transformer"
if rt_path.exists():
    sys.path.insert(0, str(rt_path))
    # RT requires maturin_import_hook for Rust modules
    # Try to install it, but continue if not available (might work if Rust module is pre-built)
    try:
        import maturin_import_hook
        from maturin_import_hook.settings import MaturinSettings
        maturin_import_hook.install(settings=MaturinSettings(release=True, uv=True))
    except ImportError:
        # maturin_import_hook not installed - might still work if Rust module is already built
        pass
    
    try:
        from rt.model import RelationalTransformer, _make_block_mask
        from rt.data import RelationalDataset
        from torch.utils.data import DataLoader
    except ImportError as e:
        raise ImportError(
            f"Failed to import RT modules. This usually means:\n"
            f"1. maturin_import_hook is not installed: pip install maturin-import-hook\n"
            f"2. Rust sampler is not built: cd relational-transformer/rustler && maturin develop --release\n"
            f"Original error: {e}"
        )
else:
    raise ImportError(f"relational-transformer not found at {rt_path}")


class RTEncoderOnly(Module):
    """
    Wrapper around RelationalTransformer that returns embeddings only
    (no decoder/prediction head).

    This replicates the encoder portion of RT's forward() method:
    - builds attention masks
    - builds block masks via _make_block_mask (which wants seq_len multiple of 128)
    - encodes cell values
    - runs relational blocks
    - applies final norm_out
    """
    def __init__(self, rt_model: RelationalTransformer):
        super().__init__()
        self.rt_model = rt_model
        # 1) Don't store KV cache during training

        
    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        Forward pass that returns cell embeddings only.
        Returns: (B, S, d_model) cell embeddings
        """
        if not hasattr(self, '_rt_forward_count'):
            self._rt_forward_count = 0
        if self._rt_forward_count == 0:
            print(f"[DEBUG] RTEncoderOnly.forward() started")
        
        node_idxs = batch["node_idxs"]
        f2p_nbr_idxs = batch["f2p_nbr_idxs"]
        col_name_idxs = batch["col_name_idxs"]
        table_name_idxs = batch["table_name_idxs"]
        is_padding = batch["is_padding"]
        batch_size, seq_len = node_idxs.shape
        device = node_idxs.device
        
        if self._rt_forward_count == 0:
            print(f"[DEBUG] RT encoder: batch_size={batch_size}, seq_len={seq_len}, device={device}")

        # ---- PAD SEQUENCE TO MULTIPLE OF 128 FOR FLEX_ATTENTION FIRST ----
        # Must pad BEFORE building attention masks and block masks
        original_seq_len = seq_len
        needs_padding = (seq_len > 0) and (seq_len % 128 != 0)

        if needs_padding:
            padded_seq_len = ((seq_len + 127) // 128) * 128
            pad_size = padded_seq_len - seq_len
            
            # Pad batch values for encoding
            padded_batch = {}
            for k, v in batch.items():
                if k == "is_padding":
                    padded_batch[k] = F.pad(v, (0, pad_size), value=True)
                elif v.dim() == 2 and v.shape[1] == original_seq_len:
                    padded_batch[k] = F.pad(v, (0, pad_size), value=0)
                elif v.dim() == 3 and v.shape[1] == original_seq_len:
                    padded_batch[k] = F.pad(v, (0, 0, 0, pad_size), value=0.0 if k.endswith("_values") else -1)
                else:
                    padded_batch[k] = v
            
            # Update batch and seq_len
            batch = padded_batch
            seq_len = padded_seq_len
            node_idxs = batch["node_idxs"]
            f2p_nbr_idxs = batch["f2p_nbr_idxs"]
            col_name_idxs = batch["col_name_idxs"]
            table_name_idxs = batch["table_name_idxs"]
            is_padding = batch["is_padding"]  # Use the padded is_padding

        # ---- BUILD ATTENTION MASKS WITH PADDED SEQUENCE LENGTH ----
        if self._rt_forward_count == 0:
            print(f"[DEBUG] RT encoder: Building attention masks...")
        # Padding mask for attention pairs (allow only non-pad -> non-pad)
        pad = (~is_padding[:, :, None]) & (~is_padding[:, None, :])  # (B, S, S)
        
        # cells in the same node
        same_node = node_idxs[:, :, None] == node_idxs[:, None, :]  # (B, S, S)
        
        # kv index is among q's foreign -> primary neighbors
        kv_in_f2p = (node_idxs[:, None, :, None] == f2p_nbr_idxs[:, :, None, :]).any(
            -1
        )  # (B, S, S)
        
        # q index is among kv's primary -> foreign neighbors (reverse relation)
        q_in_f2p = (node_idxs[:, :, None, None] == f2p_nbr_idxs[:, None, :, :]).any(
            -1
        )  # (B, S, S)
        
        # Same column AND same table
        same_col_table = (col_name_idxs[:, :, None] == col_name_idxs[:, None, :]) & (
            table_name_idxs[:, :, None] == table_name_idxs[:, None, :]
        )  # (B, S, S)
        
        # Final boolean masks (apply padding once here)
        attn_masks = {
            "feat": (same_node | kv_in_f2p) & pad,
            "nbr": q_in_f2p & pad,
            "col": same_col_table & pad,
            "full": pad,
        }
        
        # Make them contiguous for better kernel performance
        for l in attn_masks:
            attn_masks[l] = attn_masks[l].contiguous()
        
        # ---- BUILD BLOCK MASKS WITH PADDED SEQUENCE LENGTH ----
        if self._rt_forward_count == 0:
            print(f"[DEBUG] RT encoder: Building block masks...")
        # Convert to block masks (seq_len must match the actual sequence length going into blocks)
        make_block_mask = partial(
            _make_block_mask,
            batch_size=batch_size,
            seq_len=seq_len,  # Use padded seq_len, not original_seq_len
            device=device,
        )
        block_masks = {
            l: make_block_mask(attn_mask) for l, attn_mask in attn_masks.items()
        }
        if self._rt_forward_count == 0:
            print(f"[DEBUG] RT encoder: Block masks built")

        expected_S = seq_len

        # ---- ENCODE CELL VALUES ----
        x = 0
        col_name_values = batch["col_name_values"]
        if col_name_values.shape[1] != expected_S:
            raise RuntimeError(
                f"col_name_values length mismatch: expected {expected_S}, "
                f"got {col_name_values.shape[1]}"
            )

        x = x + (
            self.rt_model.norm_dict["col_name"](
                self.rt_model.enc_dict["col_name"](col_name_values)
            ) * (~is_padding)[..., None]
        )
        
        for i, t in enumerate(["number", "text", "datetime", "boolean"]):
            value_key = t + "_values"
            if value_key not in batch:
                continue
            values = batch[value_key]
            if values.shape[1] != expected_S:
                raise RuntimeError(
                    f"{value_key} length mismatch: expected {expected_S}, "
                    f"got {values.shape[1]}"
                )

            x = x + (
                self.rt_model.norm_dict[t](self.rt_model.enc_dict[t](values))
                * ((batch["sem_types"] == i) & ~batch["masks"] & ~is_padding)[..., None]
            )
            x = x + (
                self.rt_model.mask_embs[t]
                * ((batch["sem_types"] == i) & batch["masks"] & ~is_padding)[..., None]
            )
        
        # ---- RELATIONAL BLOCKS ----
        if self._rt_forward_count == 0:
            print(f"[DEBUG] RT encoder: Running {len(self.rt_model.blocks)} relational blocks...")
        for i, block in enumerate(self.rt_model.blocks):
            if self._rt_forward_count == 0 and i == 0:
                print(f"[DEBUG] RT encoder: Running block {i+1}/{len(self.rt_model.blocks)}...")
            x = block(x, block_masks)
        if self._rt_forward_count == 0:
            print(f"[DEBUG] RT encoder: All blocks complete")
        
        # ---- FINAL NORM ----
        if self._rt_forward_count == 0:
            print(f"[DEBUG] RT encoder: Applying final norm...")
        x = self.rt_model.norm_out(x)
        if self._rt_forward_count == 0:
            print(f"[DEBUG] RT encoder: Forward pass complete, output shape={x.shape}")
        self._rt_forward_count += 1

        # ---- UNPAD BACK TO ORIGINAL SEQ_LEN ----
        if needs_padding:
            x = x[:, :original_seq_len, :]  # (B, S_orig, d_model)

        return x


class HeteroDataToRTBatch:
    """
    Converts Rel-LLM's HeteroData format to RT's batch format.
    """
    def __init__(self, col_stats_dict: Dict, text_embedder=None, d_text: int = 384):
        self.col_stats_dict = col_stats_dict
        self.text_embedder = text_embedder
        self.d_text = d_text
        # Cache for column/table name mappings
        self.col_name_to_idx = {}
        self.table_name_to_idx = {}
        self._mapping_built = False
        
    def _build_mappings(self, batch: HeteroData):
        """Build column and table name to index mappings."""
        if self._mapping_built:
            return
            
        # Build table name mapping
        for idx, node_type in enumerate(batch.node_types):
            self.table_name_to_idx[node_type] = idx
        
        # Build column name mapping
        col_idx = 0
        for node_type in batch.node_types:
            tf = batch.tf_dict[node_type]
            for stype_name, col_names in tf.col_names_dict.items():
                for col_name in col_names:
                    key = f"{col_name} of {node_type}"
                    if key not in self.col_name_to_idx:
                        self.col_name_to_idx[key] = col_idx
                        col_idx += 1
        
        self._mapping_built = True
    
    def _get_col_idx(self, col_name: str, table_name: str) -> int:
        """Get column name index."""
        key = f"{col_name} of {table_name}"
        return self.col_name_to_idx.get(key, 0)
    
    def _get_table_idx(self, table_name: str) -> int:
        """Get table name index."""
        return self.table_name_to_idx.get(table_name, 0)
    
    def _get_text_embedding(self, text_value) -> np.ndarray:
        """Get text embedding for a cell value. Always returns numpy array of length self.d_text."""
        if self.text_embedder is not None:
            try:
                emb = self.text_embedder([str(text_value)])
                if isinstance(emb, Tensor):
                    emb = emb.detach().cpu().numpy()
                if isinstance(emb, np.ndarray):
                    result = emb[0] if emb.ndim > 1 else emb
                elif isinstance(emb, list):
                    result = emb[0]
                else:
                    result = emb

                if isinstance(result, Tensor):
                    result = result.detach().cpu().numpy()

                vec = np.asarray(result, dtype=np.float32)

                d = vec.shape[-1]
                if d < self.d_text:
                    vec = np.pad(vec, (0, self.d_text - d), mode="constant")
                elif d > self.d_text:
                    vec = vec[: self.d_text]

                return vec
            except Exception:
                pass

        # Fallback
        return np.zeros(self.d_text, dtype=np.float32)

    
    def _compute_f2p_neighbors(
        self, 
        batch: HeteroData, 
        node_idxs_list: List[int],
        table_name_idxs_list: List[int]
    ) -> Tensor:
        """
        Compute foreign-to-primary neighbor indices from edge_index_dict.
        
        Returns:
            f2p_nbr_idxs: (1, S, 5) - max 5 neighbors per cell
        """
        # Get device from any available tensor in the batch
        device = None
        for node_type in batch.node_types:
            if hasattr(batch[node_type], 'x') and batch[node_type].x is not None:
                device = batch[node_type].x.device
                break
            if hasattr(batch[node_type], 'seed_time') and batch[node_type].seed_time is not None:
                device = batch[node_type].seed_time.device
                break
            if hasattr(batch[node_type], 'time') and batch[node_type].time is not None:
                device = batch[node_type].time.device
                break
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        seq_len = len(node_idxs_list)
        max_neighbors = 5
        
        # Initialize with -1 (no neighbor)
        f2p_nbr_idxs = torch.full((1, seq_len, max_neighbors), -1, dtype=torch.long, device=device)
        
        # Build mapping from (table_idx, node_idx) -> cell_idx
        table_node_to_cell = {}
        for cell_idx, (node_idx, table_idx) in enumerate(zip(node_idxs_list, table_name_idxs_list)):
            key = (table_idx.item() if isinstance(table_idx, Tensor) else table_idx,
                   node_idx.item() if isinstance(node_idx, Tensor) else node_idx)
            if key not in table_node_to_cell:
                table_node_to_cell[key] = []
            table_node_to_cell[key].append(cell_idx)
        
        # Process edges to find neighbors
        for edge_type in batch.edge_types:
            src_type, _, dst_type = edge_type
            src_idx = self._get_table_idx(src_type)
            dst_idx = self._get_table_idx(dst_type)
            
            if edge_type in batch.edge_index_dict:
                edge_index = batch.edge_index_dict[edge_type]
                src_nodes = edge_index[0].cpu().numpy()
                dst_nodes = edge_index[1].cpu().numpy()
                
                # For each edge, find corresponding cells and link them
                for src_node, dst_node in zip(src_nodes, dst_nodes):
                    src_key = (src_idx, src_node)
                    dst_key = (dst_idx, dst_node)
                    
                    if src_key in table_node_to_cell and dst_key in table_node_to_cell:
                        src_cells = table_node_to_cell[src_key]
                        dst_cells = table_node_to_cell[dst_key]
                        
                        # Link src cells to dst cells (F->P relationship)
                        for src_cell_idx in src_cells:
                            neighbor_count = 0
                            for dst_cell_idx in dst_cells[:max_neighbors]:
                                if neighbor_count < max_neighbors:
                                    f2p_nbr_idxs[0, src_cell_idx, neighbor_count] = dst_cell_idx
                                    neighbor_count += 1
        
        return f2p_nbr_idxs
    
    
    def convert(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        batch_size: int,
    ) -> Dict[str, Tensor]:
        """
        Convert HeteroData batch to RT batch format.
        
        * node_idxs are LOCAL per table: 0 .. num_nodes(table)-1
        * all *_values have shape (1, S, ...)
        """
        self._build_mappings(batch)
        device = batch[entity_table].seed_time.device
        
        all_cells: List[dict] = []
        node_idxs_list: List[int] = []
        col_name_idxs_list: List[int] = []
        table_name_idxs_list: List[int] = []
        sem_types_list: List[int] = []

        # ---- First pass: enumerate all cells ----
        for node_type in batch.node_types:
            tf = batch.tf_dict[node_type]
            num_nodes = len(tf)
            table_idx = self._get_table_idx(node_type)
            
            for node_idx_in_table in range(num_nodes):
                # node_idx is LOCAL to this table
                node_idx = node_idx_in_table
                
                for stype_name, feat in tf.feat_dict.items():
                    col_names = tf.col_names_dict.get(stype_name, [])
                    
                    if stype_name == stype.numerical:
                        if isinstance(feat, Tensor):
                            for col_j, col_name in enumerate(col_names):
                                value = feat[node_idx_in_table, col_j].item()
                                all_cells.append({
                                    "value": value,
                                    "sem_type": 0,  # number
                                    "node_idx": node_idx,
                                    "col_name": col_name,
                                    "table_name": node_type,
                                })
                                node_idxs_list.append(node_idx)
                                col_name_idxs_list.append(self._get_col_idx(col_name, node_type))
                                table_name_idxs_list.append(table_idx)
                                sem_types_list.append(0)
                    
                    elif stype_name == stype.categorical:
                        if isinstance(feat, Tensor):
                            for col_j, col_name in enumerate(col_names):
                                value = feat[node_idx_in_table, col_j].item()
                                all_cells.append({
                                    "value": float(value),
                                    "sem_type": 0,  # treat as number
                                    "node_idx": node_idx,
                                    "col_name": col_name,
                                    "table_name": node_type,
                                })
                                node_idxs_list.append(node_idx)
                                col_name_idxs_list.append(self._get_col_idx(col_name, node_type))
                                table_name_idxs_list.append(table_idx)
                                sem_types_list.append(0)
                    
                    elif stype_name == stype.text_embedded:
                        if isinstance(feat, torch_frame.data.MultiEmbeddingTensor):
                            offset = feat.offset
                            for col_j, col_name in enumerate(col_names):
                                start_idx = offset[node_idx_in_table * len(col_names) + col_j]
                                end_idx = offset[node_idx_in_table * len(col_names) + col_j + 1]
                                value = feat.values[start_idx:end_idx].cpu().numpy()
                                
                                all_cells.append({
                                    "value": value,
                                    "sem_type": 1,  # text
                                    "node_idx": node_idx,
                                    "col_name": col_name,
                                    "table_name": node_type,
                                })
                                node_idxs_list.append(node_idx)
                                col_name_idxs_list.append(self._get_col_idx(col_name, node_type))
                                table_name_idxs_list.append(table_idx)
                                sem_types_list.append(1)
                    
                    elif stype_name == stype.timestamp:
                        # Timestamp -> scalar datetime
                        if isinstance(feat, Tensor):
                            for col_j, col_name in enumerate(col_names):
                                feat_val = feat[node_idx_in_table, col_j]
                                if feat_val.dim() == 0:
                                    value = feat_val.item()
                                else:
                                    comps = feat_val.float()
                                    year = comps[0].item() if comps.numel() > 0 else 2000
                                    month = comps[1].item() if comps.numel() > 1 else 1
                                    day = comps[2].item() if comps.numel() > 2 else 1
                                    value = (year - 2000) * 365 + month * 30 + day

                                all_cells.append({
                                    "value": value,
                                    "sem_type": 2,  # datetime
                                    "node_idx": node_idx,
                                    "col_name": col_name,
                                    "table_name": node_type,
                                })
                                node_idxs_list.append(node_idx)
                                col_name_idxs_list.append(self._get_col_idx(col_name, node_type))
                                table_name_idxs_list.append(table_idx)
                                sem_types_list.append(2)
                
        seq_len = len(all_cells)
        
        # ---- Second pass: build aligned value arrays (length = seq_len) ----
        number_rows:   List[np.ndarray] = []
        text_rows:     List[np.ndarray] = []
        datetime_rows: List[np.ndarray] = []
        boolean_rows:  List[np.ndarray] = []
        colname_rows:  List[np.ndarray] = []
        
        for cell in all_cells:
            colname_rows.append(self._get_text_embedding(cell["col_name"]))  # always filled

            st = cell["sem_type"]
            v  = cell["value"]

            if st == 0:  # number
                number_rows.append(np.array([float(v)], dtype=np.float32))
                text_rows.append(np.zeros(self.d_text, dtype=np.float32))
                datetime_rows.append(np.zeros(1, dtype=np.float32))
                boolean_rows.append(np.zeros(1, dtype=np.float32))

            elif st == 1:  # text
                if isinstance(v, np.ndarray):
                    vec = v.astype(np.float32)
                else:
                    vec = self._get_text_embedding(v)
                # ensure length d_text
                if vec.shape[-1] < self.d_text:
                    vec = np.pad(vec, (0, self.d_text - vec.shape[-1]), mode="constant")
                elif vec.shape[-1] > self.d_text:
                    vec = vec[: self.d_text]
                number_rows.append(np.zeros(1, dtype=np.float32))
                text_rows.append(vec)
                datetime_rows.append(np.zeros(1, dtype=np.float32))
                boolean_rows.append(np.zeros(1, dtype=np.float32))

            elif st == 2:  # datetime
                number_rows.append(np.zeros(1, dtype=np.float32))
                text_rows.append(np.zeros(self.d_text, dtype=np.float32))
                datetime_rows.append(np.array([float(v)], dtype=np.float32))
                boolean_rows.append(np.zeros(1, dtype=np.float32))

            elif st == 3:  # boolean
                number_rows.append(np.zeros(1, dtype=np.float32))
                text_rows.append(np.zeros(self.d_text, dtype=np.float32))
                datetime_rows.append(np.zeros(1, dtype=np.float32))
                boolean_rows.append(np.array([float(v > 0)], dtype=np.float32))

        else:
                # Fallback: everything zero
                number_rows.append(np.zeros(1, dtype=np.float32))
                text_rows.append(np.zeros(self.d_text, dtype=np.float32))
                datetime_rows.append(np.zeros(1, dtype=np.float32))
                boolean_rows.append(np.zeros(1, dtype=np.float32))

        # Stack to numpy arrays, then to tensors (B=1)
        number_arr   = np.stack(number_rows,   axis=0)   # (S, 1)
        text_arr     = np.stack(text_rows,     axis=0)   # (S, d_text)
        datetime_arr = np.stack(datetime_rows, axis=0)   # (S, 1)
        boolean_arr  = np.stack(boolean_rows,  axis=0)   # (S, 1)
        colname_arr  = np.stack(colname_rows,  axis=0)   # (S, d_text)

        rt_batch = {
            "node_idxs":       torch.tensor(node_idxs_list,       device=device, dtype=torch.long).unsqueeze(0),
            "col_name_idxs":   torch.tensor(col_name_idxs_list,   device=device, dtype=torch.long).unsqueeze(0),
            "table_name_idxs": torch.tensor(table_name_idxs_list, device=device, dtype=torch.long).unsqueeze(0),
            "sem_types":       torch.tensor(sem_types_list,       device=device, dtype=torch.long).unsqueeze(0),
            "is_padding":      torch.zeros(1, seq_len, dtype=torch.bool, device=device),
            "masks":           torch.zeros(1, seq_len, dtype=torch.bool, device=device),
            "number_values":   torch.tensor(number_arr,   device=device, dtype=torch.float32).unsqueeze(0),
            "text_values":     torch.tensor(text_arr,     device=device, dtype=torch.float32).unsqueeze(0),
            "datetime_values": torch.tensor(datetime_arr, device=device, dtype=torch.float32).unsqueeze(0),
            "boolean_values":  torch.tensor(boolean_arr,  device=device, dtype=torch.float32).unsqueeze(0),
            "col_name_values": torch.tensor(colname_arr,  device=device, dtype=torch.float32).unsqueeze(0),
        }
        
        # f2p_nbr_idxs uses LOCAL node indices (0..num_nodes-1) per table
        rt_batch["f2p_nbr_idxs"] = self._compute_f2p_neighbors(
            batch, node_idxs_list, table_name_idxs_list
        )
        
        return rt_batch


def aggregate_cells_to_nodes(
    cell_embeds: Tensor,  # (B, S, d_model) - may be unpadded
    rt_batch: Dict[str, Tensor],
    batch: HeteroData,
    entity_table: NodeType,
    aggr: str = "sum"
) -> Dict[NodeType, Tensor]:
    device = cell_embeds.device
    B, S, d_model = cell_embeds.shape
    assert B == 1, f"Expected batch size 1, got {B}"
    cell_embeds = cell_embeds[0]  # (S_unpadded, d_model)

    node_idxs = rt_batch['node_idxs'][0]          # (S_padded,)
    table_name_idxs = rt_batch['table_name_idxs'][0]  # (S_padded,)

    # Unpad indices to match cell_embeds length
    actual_seq_len = cell_embeds.shape[0]
    if node_idxs.shape[0] > actual_seq_len:
        node_idxs = node_idxs[:actual_seq_len]
        table_name_idxs = table_name_idxs[:actual_seq_len]

    x_dict: Dict[NodeType, Tensor] = {}

    for table_idx, node_type in enumerate(batch.node_types):
        tf = batch.tf_dict[node_type]
        num_nodes = len(tf)  # number of nodes for this type in this batch

        # Select cells of this table
        mask = (table_name_idxs == table_idx)
        if not mask.any():
            # No cells at all for this table -> all-zero embeddings, but keep num_nodes
            x_dict[node_type] = torch.zeros(num_nodes, d_model, device=device)
            continue

        table_node_idxs = node_idxs[mask]      # (S_table,)
        table_cell_embeds = cell_embeds[mask]  # (S_table, d_model)

        # Sanity: local node indices must be in [0, num_nodes)
        if table_node_idxs.numel() > 0:
            max_idx = int(table_node_idxs.max().item())
            if max_idx >= num_nodes:
                raise RuntimeError(
                    f"Node index out of range for node_type={node_type}: "
                    f"max_idx={max_idx}, num_nodes={num_nodes}"
                )

        # Initialize one embedding per node
        node_embeds = torch.zeros(num_nodes, d_model, device=device)

        if aggr in ("sum", "mean"):
            counts = torch.zeros(num_nodes, dtype=torch.long, device=device)

            for emb, n_idx in zip(table_cell_embeds, table_node_idxs):
                idx = int(n_idx.item())
                node_embeds[idx] += emb
                counts[idx] += 1

            if aggr == "mean":
                mask_nonzero = counts > 0
                node_embeds[mask_nonzero] = (
                    node_embeds[mask_nonzero] / counts[mask_nonzero].unsqueeze(-1)
                )

        elif aggr == "max":
            counts = torch.zeros(num_nodes, dtype=torch.long, device=device)
            node_embeds[:] = -1e9
            for emb, n_idx in zip(table_cell_embeds, table_node_idxs):
                idx = int(n_idx.item())
                node_embeds[idx] = torch.maximum(node_embeds[idx], emb)
                counts[idx] += 1
            # Nodes with no cells -> set back to zero
            mask_zero = counts == 0
            node_embeds[mask_zero] = 0.0
        else:
            raise ValueError(f"Unknown aggregation method: {aggr}")

        x_dict[node_type] = node_embeds

    return x_dict


def aggregate_cells_to_nodes_from_rt_batch(
    cell_embeds: Tensor,  # (B, S, d_model)
    rt_batch: Dict[str, Tensor],
    table_idx_to_node_type: Dict[int, str],
    aggr: str = "sum"
) -> Dict[str, Tensor]:
    """
    Aggregate cell embeddings to node embeddings from RT batch format.
    Works without HeteroData - uses RT's table indices.
    
    Args:
        cell_embeds: (B, S, d_model) cell embeddings from RT encoder
        rt_batch: RT batch dict with node_idxs, table_name_idxs
        table_idx_to_node_type: Mapping from RT table index to node type name
        aggr: Aggregation method ("sum", "mean", "max")
        
    Returns:
        x_dict: Dict mapping node_type -> (num_nodes, d_model) embeddings
    """
    device = cell_embeds.device
    B, S, d_model = cell_embeds.shape
    
    # Flatten batch dimension: (B, S, d_model) -> (B*S, d_model)
    cell_embeds_flat = cell_embeds.view(-1, d_model)  # (B*S, d_model)

    node_idxs = rt_batch['node_idxs'].flatten()  # (B*S,) - RT's global node indices
    table_name_idxs = rt_batch['table_name_idxs'].flatten()  # (B*S,)

    x_dict: Dict[str, Tensor] = {}

    # Group by table
    for table_idx_tensor in torch.unique(table_name_idxs):
        table_idx = int(table_idx_tensor.item())
        if table_idx not in table_idx_to_node_type:
            continue
        
        node_type = table_idx_to_node_type[table_idx]
        mask = (table_name_idxs == table_idx)
        
        if not mask.any():
            continue
        
        table_node_idxs = node_idxs[mask]  # Global node indices
        table_cell_embeds = cell_embeds_flat[mask]  # (S_table, d_model)

        # Get unique nodes and their local indices
        unique_nodes, inverse_indices = torch.unique(table_node_idxs, return_inverse=True)
        num_nodes = len(unique_nodes)

        # Initialize node embeddings
        node_embeds = torch.zeros(num_nodes, d_model, device=device)

        if aggr in ("sum", "mean"):
            counts = torch.zeros(num_nodes, dtype=torch.long, device=device)
            
            for emb, local_idx in zip(table_cell_embeds, inverse_indices):
                node_embeds[local_idx] += emb
                counts[local_idx] += 1

            if aggr == "mean":
                mask_nonzero = counts > 0
                node_embeds[mask_nonzero] = (
                    node_embeds[mask_nonzero] / counts[mask_nonzero].unsqueeze(-1)
                )

        elif aggr == "max":
            counts = torch.zeros(num_nodes, dtype=torch.long, device=device)
            node_embeds[:] = -1e9
            for emb, local_idx in zip(table_cell_embeds, inverse_indices):
                node_embeds[local_idx] = torch.maximum(node_embeds[local_idx], emb)
                counts[local_idx] += 1
            mask_zero = counts == 0
            node_embeds[mask_zero] = 0.0
        else:
            raise ValueError(f"Unknown aggregation method: {aggr}")

        x_dict[node_type] = node_embeds

    return x_dict


# ============================================================================
# RT SAMPLER BRIDGE - New components for using RT's sampler directly
# ============================================================================


class RTBatchToRelLLMFormat:
    """
    Bridge that converts RT batch format + metadata → Rel-LLM format.
    
    RT's sampler returns batches in RT format, but Rel-LLM needs:
    - seed_time: timestamps for seed nodes
    - n_id: global node IDs  
    - time_dict: timestamps for all nodes
    - batch_dict: mapping nodes to seed nodes
    """
    def __init__(self, task, entity_table: str, dataset_name: str):
        self.task = task
        self.entity_table = entity_table
        self.dataset_name = dataset_name
        
        # RT uses task table name (e.g., "user-churn") while Rel-LLM uses entity table name (e.g., "customer").
        # Default to task.name, but we may override after loading table_info if entity_table exists there.
        self.rt_task_table_name = task.name  # e.g., "user-churn" for user-churn task
        
        # Load RT's table_info to map node indices and table names
        home = os.environ.get("HOME", ".")
        table_info_path = os.path.join(home, "scratch", "pre", dataset_name, "table_info.json")
        if os.path.exists(table_info_path):
            with open(table_info_path) as f:
                self.table_info = json.load(f)
        else:
            self.table_info = {}
        
        # Build mapping from RT table_name_idx → table name (node type).
        # RT's table_name_idx comes from text_to_idx in preprocessing (text.json),
        # not from node_idx_offset ordering.
        self.table_idx_to_node_type = {}
        self.node_type_to_table_idx = {}

        # Load text.json to recover text_to_idx mapping used by Rust preprocessor
        text_json_path = os.path.join(home, "scratch", "pre", dataset_name, "text.json")
        text_to_idx = {}
        if os.path.exists(text_json_path):
            try:
                with open(text_json_path) as f:
                    text_list = json.load(f)
                # Build str -> idx
                text_to_idx = {s: i for i, s in enumerate(text_list)}
            except Exception:
                text_to_idx = {}

        # Table names are in table_info keys (table_name:TableType)
        table_names = set()
        for key in self.table_info.keys():
            table_names.add(key.split(":")[0])

        # Use text_to_idx if available (correct mapping), otherwise fallback
        if text_to_idx:
            for table_name in table_names:
                if table_name in text_to_idx:
                    table_idx = int(text_to_idx[table_name])
                    self.table_idx_to_node_type[table_idx] = table_name
                    if table_name not in self.node_type_to_table_idx:
                        self.node_type_to_table_idx[table_name] = table_idx
        else:
            # Fallback: previous heuristic based on node_idx_offset ordering
            table_entries = []
            for key, info in self.table_info.items():
                table_name = key.split(":")[0]
                table_entries.append((table_name, info["node_idx_offset"], key))
            table_entries.sort(key=lambda x: x[1])
            for table_idx, (table_name, _, _) in enumerate(table_entries):
                self.table_idx_to_node_type[table_idx] = table_name
                if table_name not in self.node_type_to_table_idx:
                    self.node_type_to_table_idx[table_name] = table_idx
        
        # If the task name doesn't exist in RT's table_info but the entity_table does,
        # fall back to entity_table as the RT task table name.
        if (
            self.rt_task_table_name not in self.node_type_to_table_idx
            and self.entity_table in self.node_type_to_table_idx
        ):
            self.rt_task_table_name = self.entity_table

        # Build reverse mapping: RT task table name → Rel-LLM entity table name
        # For the entity table, use the resolved RT task table name
        self.rt_to_relllm_table_map = {self.rt_task_table_name: self.entity_table}
    
    def extract_seed_nodes(self, rt_batch: Dict[str, Tensor]) -> Tuple[List[Tensor], int]:
        """
        Extract seed node information from RT batch.
        
        Returns:
            seed_cell_indices_list: List of (num_seeds_i,) tensors, one per batch item
            batch_size: total number of seed nodes across all batch items
        """
        # RT batch has is_task_nodes: (B, S) marking seed nodes
        is_task_nodes = rt_batch['is_task_nodes']  # (B, S)
        B, S = is_task_nodes.shape
        
        node_idxs = rt_batch['node_idxs']  # (B, S)
        table_name_idxs = rt_batch['table_name_idxs']  # (B, S)
        
        # Extract seed nodes for each batch item
        seed_cell_indices_list = []
        total_seed_nodes = 0
        
        for b in range(B):
            seed_mask = is_task_nodes[b]  # (S,)
            seed_cell_indices = torch.where(seed_mask)[0]  # Indices of seed cells for this batch item
            
            # Get unique seed nodes (multiple cells per node)
            batch_node_idxs = node_idxs[b]  # (S,)
            batch_table_name_idxs = table_name_idxs[b]  # (S,)
            
            seed_nodes = set()
            for cell_idx in seed_cell_indices:
                cell_idx_int = int(cell_idx.item())
                table_idx = int(batch_table_name_idxs[cell_idx_int].item())
                node_idx = int(batch_node_idxs[cell_idx_int].item())
                seed_nodes.add((table_idx, node_idx))
            
            seed_cell_indices_list.append(seed_cell_indices)
            total_seed_nodes += len(seed_nodes)
        
        return seed_cell_indices_list, total_seed_nodes
    
    def extract_seed_time(self, split: str) -> Tensor:
        """
        Extract seed_time from task table.
        """
        table = self.task.get_table(split)
        # Get timestamp column (usually 'time' or task-specific)
        if hasattr(self.task, 'time_col'):
            time_col = self.task.time_col
        else:
            # Try common names
            time_col = None
            for col in ['time', 'timestamp', 't']:
                if col in table.df.columns:
                    time_col = col
                    break
            if time_col is None:
                raise ValueError(f"Could not find timestamp column in {split} table")
        
        seed_times = torch.from_numpy(table.df[time_col].values.astype(np.int64))
        return seed_times
    
    def build_metadata(
        self, 
        rt_batch: Dict[str, Tensor],
        split: str
    ) -> Dict:
        """
        Build Rel-LLM metadata from RT batch (OPTIMIZED VERSION).
        
        For RT batches, we skip expensive nested loops since:
        - Temporal encoder is skipped (time_dict=None)
        - batch_dict is only used for temporal encoder
        - n_id_dict is only needed for ID embeddings (optional)
        
        Returns dict with:
        - seed_time: (batch_size,) timestamps for seed nodes
        - n_id: dict mapping node_type -> global node IDs (minimal, for ID embeddings if needed)
        - time_dict: empty dict (temporal encoder skipped for RT)
        - batch_dict: empty dict (not used when temporal encoder skipped)
        """
        # Extract seed nodes (needed for batch_size)
        seed_cell_indices_list, batch_size = self.extract_seed_nodes(rt_batch)
        
        # Cheap seed_time: just slice table times
        all_seed_times = self.extract_seed_time(split)
        seed_time = all_seed_times[:batch_size] if len(all_seed_times) >= batch_size else all_seed_times
        
        # Build minimal n_id_dict: only for nodes that appear in batch (for ID embeddings if enabled)
        # This is much cheaper than the full nested loop version
        node_idxs = rt_batch['node_idxs']  # (B, S)
        table_name_idxs = rt_batch['table_name_idxs']  # (B, S)
        B, S = node_idxs.shape
        
        n_id_dict = {}
        # Group cells by table - vectorized version
        all_table_idxs = torch.unique(table_name_idxs.flatten())
        for table_idx_tensor in all_table_idxs:
            table_idx = int(table_idx_tensor.item())
            if table_idx not in self.table_idx_to_node_type:
                continue
            node_type = self.table_idx_to_node_type[table_idx]
            
            # Vectorized: collect all nodes for this table across all batch items
            mask = (table_name_idxs.flatten() == table_idx)
            table_node_idxs = node_idxs.flatten()[mask]
            if len(table_node_idxs) > 0:
                unique_nodes = torch.unique(table_node_idxs)
                n_id_dict[node_type] = unique_nodes
            else:
                n_id_dict[node_type] = torch.tensor([], dtype=torch.long, device=node_idxs.device)
        
        # For RT batches, we don't need batch_dict or time_dict
        # Temporal encoder is skipped, and batch_dict is only used for temporal encoding
        time_dict = {}  # Empty - temporal encoder skips None/empty dict
        batch_dict = {}  # Empty - not used when temporal encoder skipped
        
        return {
            'seed_time': seed_time,
            'n_id': n_id_dict,
            'time_dict': time_dict,
            'batch_dict': batch_dict,
            'batch_size': batch_size
        }


def create_rt_loader(
    task,
    dataset_name: str,
    entity_table: str,
    split: str,
    batch_size: int = 32,
    seq_len: int = 1024,
    max_bfs_width: int = 256,
    embedding_model: str = "all-MiniLM-L12-v2",
    d_text: int = 384,
    num_workers: int = 0,
    seed: int = 0,
):
    """
    Create RT's DataLoader using RelationalDataset.
    
    Returns:
        loader: DataLoader that yields RT batch format
        bridge: RTBatchToRelLLMFormat for converting to Rel-LLM format
    """
    # Get target column name
    target_column = task.target_col if hasattr(task, 'target_col') else task.name.split('-')[-1]
    
    # RT expects the task table name (e.g., "user-churn"), not the entity table (e.g., "customer")
    # The task table name is typically the task name itself
    task_table_name = task.name  # e.g., "user-churn"
    
    # Create RT dataset
    print(f"[DEBUG] Creating RT RelationalDataset for {split} split...")
    rt_dataset = RelationalDataset(
        tasks=[(dataset_name, task_table_name, target_column, split, [])],
        batch_size=batch_size,
        seq_len=seq_len,
        rank=0,  # Single GPU
        world_size=1,
        max_bfs_width=max_bfs_width,
        embedding_model=embedding_model,
        d_text=d_text,
        seed=seed,
    )
    print(f"[DEBUG] RT dataset created, length={len(rt_dataset)}")
    
    # Create DataLoader
    print(f"[DEBUG] Creating DataLoader with num_workers={num_workers}...")
    loader = DataLoader(
        rt_dataset,
        batch_size=None,  # Dataset returns batches directly
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        shuffle=(split == "train"),
    )
    print(f"[DEBUG] DataLoader created for {split} split")
    
    # Create bridge
    bridge = RTBatchToRelLLMFormat(task, entity_table, dataset_name)
    
    return loader, bridge
