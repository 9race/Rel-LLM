"""
Adapter module to integrate relational-transformer repository with Rel-LLM.
Reuses existing RT code without rewriting.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType
import torch_frame
from torch_frame import stype
import numpy as np

# Add relational-transformer to path
rt_path = Path(__file__).parent / "relational-transformer"
if rt_path.exists():
    sys.path.insert(0, str(rt_path))
    from rt.model import RelationalTransformer, _make_block_mask
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
        # 1) Don’t store KV cache during training


    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        Forward pass that returns cell embeddings only.

        batch keys (before padding):
            node_idxs:       (B, S)
            f2p_nbr_idxs:    (B, S, 5)
            col_name_idxs:   (B, S)
            table_name_idxs: (B, S)
            sem_types:       (B, S)
            is_padding:      (B, S)
            masks:           (B, S)
            number_values:   (B, S, 1)
            text_values:     (B, S, d_text)
            datetime_values: (B, S, 1)
            boolean_values:  (B, S, 1)
            col_name_values: (B, S, d_text)

        Returns:
            x: (B, S, d_model) encoder output (unpadded)
        """
        node_idxs = batch["node_idxs"]
        f2p_nbr_idxs = batch["f2p_nbr_idxs"]
        col_name_idxs = batch["col_name_idxs"]
        table_name_idxs = batch["table_name_idxs"]
        is_padding = batch["is_padding"]

        batch_size, seq_len = node_idxs.shape
        device = node_idxs.device

        # Handle empty sequences
        if seq_len == 0:
            d_model = self.rt_model.d_model
            return torch.zeros(batch_size, 0, d_model, device=device)

        # ---- PAD SEQUENCE TO MULTIPLE OF 128 FOR FLEX_ATTENTION ----
        original_seq_len = seq_len
        needs_padding = (seq_len > 0) and (seq_len % 128 != 0)

        if needs_padding:
            padded_seq_len = ((seq_len + 127) // 128) * 128
            pad_size = padded_seq_len - seq_len

            padded_is_padding = F.pad(is_padding, (0, pad_size), value=True)

            padded_batch = {}
            for k, v in batch.items():
                if k == "is_padding":
                    padded_batch[k] = padded_is_padding
                elif k in ["node_idxs", "col_name_idxs", "table_name_idxs",
                           "sem_types", "masks"]:
                    # (B, S)
                    if v.dim() == 2:
                        padded_batch[k] = F.pad(v, (0, pad_size), value=0)
                    else:
                        padded_batch[k] = v
                elif k == "f2p_nbr_idxs":
                    # (B, S, 5)  pad S dimension
                    padded_batch[k] = F.pad(v, (0, 0, 0, pad_size), value=-1)
                elif k in ["number_values", "datetime_values", "boolean_values"]:
                    # (B, S, 1)
                    padded_batch[k] = F.pad(v, (0, 0, 0, pad_size), value=0.0)
                elif k in ["text_values", "col_name_values"]:
                    # (B, S, d_text)
                    padded_batch[k] = F.pad(v, (0, 0, 0, pad_size), value=0.0)
                else:
                    # any other tensors untouched
                    padded_batch[k] = v

            # Sanity checks: everything that depends on S should now be padded_seq_len
            for k in ["number_values", "text_values", "datetime_values",
                      "boolean_values", "col_name_values"]:
                if k in padded_batch:
                    v = padded_batch[k]
                    if v.dim() >= 2:
                        if v.shape[1] != padded_seq_len:
                            raise RuntimeError(
                                f"Padding mismatch for {k}: expected S={padded_seq_len}, "
                                f"got {v.shape[1]}, shape={v.shape}"
                            )

            for k in ["is_padding", "sem_types", "masks", "node_idxs",
                      "col_name_idxs", "table_name_idxs"]:
                if k in padded_batch:
                    v = padded_batch[k]
                    if v.dim() >= 2 and v.shape[1] != padded_seq_len:
                        raise RuntimeError(
                            f"Padding mismatch for {k}: expected S={padded_seq_len}, "
                            f"got {v.shape[1]}, shape={v.shape}"
                        )

            # Update references
            batch = padded_batch
            seq_len = padded_seq_len
            is_padding = padded_is_padding
            node_idxs = batch["node_idxs"]
            f2p_nbr_idxs = batch["f2p_nbr_idxs"]
            col_name_idxs = batch["col_name_idxs"]
            table_name_idxs = batch["table_name_idxs"]

        # ---- ATTENTION MASKS (now using padded seq_len if we padded) ----
        pad = (~is_padding[:, :, None]) & (~is_padding[:, None, :])  # (B, S, S)

        same_node = node_idxs[:, :, None] == node_idxs[:, None, :]   # (B, S, S)

        kv_in_f2p = (
            node_idxs[:, None, :, None] == f2p_nbr_idxs[:, :, None, :]
        ).any(-1)                                                    # (B, S, S)

        q_in_f2p = (
            node_idxs[:, :, None, None] == f2p_nbr_idxs[:, None, :, :]
        ).any(-1)                                                    # (B, S, S)

        same_col_table = (
            (col_name_idxs[:, :, None] == col_name_idxs[:, None, :]) &
            (table_name_idxs[:, :, None] == table_name_idxs[:, None, :])
        )                                                            # (B, S, S)

        attn_masks = {
            "feat": (same_node | kv_in_f2p) & pad,
            "nbr":  q_in_f2p & pad,
            "col":  same_col_table & pad,
            "full": pad,
        }

        for k in attn_masks:
            attn_masks[k] = attn_masks[k].contiguous()

        # ---- BLOCK MASKS (flex-attention) ----
        from functools import partial
        make_block_mask = partial(
            _make_block_mask,
            batch_size=batch_size,
            seq_len=seq_len,   # NOTE: padded seq_len if needs_padding
            device=device,
        )
        block_masks = {k: make_block_mask(m) for k, m in attn_masks.items()}

        # ---- ENCODING (same as RT) ----
        expected_S = seq_len
        if batch["sem_types"].shape[1] != expected_S:
            raise RuntimeError(
                f"sem_types length mismatch: expected {expected_S}, "
                f"got {batch['sem_types'].shape[1]}"
            )
        if batch["masks"].shape[1] != expected_S:
            raise RuntimeError(
                f"masks length mismatch: expected {expected_S}, "
                f"got {batch['masks'].shape[1]}"
            )

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
        for block in self.rt_model.blocks:
            x = block(x, block_masks)

        # ---- FINAL NORM ----
        x = self.rt_model.norm_out(x)

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

        # ---- Handle empty graph ----
        seq_len = len(all_cells)
        if seq_len == 0:
            return {
                "node_idxs":       torch.zeros(1, 0, dtype=torch.long, device=device),
                "col_name_idxs":   torch.zeros(1, 0, dtype=torch.long, device=device),
                "table_name_idxs": torch.zeros(1, 0, dtype=torch.long, device=device),
                "sem_types":       torch.zeros(1, 0, dtype=torch.long, device=device),
                "is_padding":      torch.ones(1, 0, dtype=torch.bool, device=device),
                "masks":           torch.zeros(1, 0, dtype=torch.bool, device=device),
                "f2p_nbr_idxs":    torch.full((1, 0, 5), -1, dtype=torch.long, device=device),
                "number_values":   torch.zeros(1, 0, 1, device=device),
                "text_values":     torch.zeros(1, 0, self.d_text, device=device),
                "datetime_values": torch.zeros(1, 0, 1, device=device),
                "boolean_values":  torch.zeros(1, 0, 1, device=device),
                "col_name_values": torch.zeros(1, 0, self.d_text, device=device),
            }

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


