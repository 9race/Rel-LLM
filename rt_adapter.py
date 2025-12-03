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
    """
    def __init__(self, rt_model: RelationalTransformer):
        super().__init__()
        self.rt_model = rt_model
        
    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        Forward pass that returns cell embeddings only.
        
        Args:
            batch: RT batch format dict with:
                - node_idxs: (B, S)
                - f2p_nbr_idxs: (B, S, 5)
                - col_name_idxs: (B, S)
                - table_name_idxs: (B, S)
                - sem_types: (B, S)
                - is_padding: (B, S)
                - masks: (B, S)
                - number_values, text_values, datetime_values, boolean_values, col_name_values
            
        Returns:
            cell_embeds: (B, S, d_model) cell embeddings
        """
        node_idxs = batch["node_idxs"]
        f2p_nbr_idxs = batch["f2p_nbr_idxs"]
        col_name_idxs = batch["col_name_idxs"]
        table_name_idxs = batch["table_name_idxs"]
        is_padding = batch["is_padding"]
        batch_size, seq_len = node_idxs.shape
        device = node_idxs.device
        
        # Compute attention masks (same as RT)
        pad = (~is_padding[:, :, None]) & (~is_padding[:, None, :])  # (B, S, S)
        
        # Cells in the same node
        same_node = node_idxs[:, :, None] == node_idxs[:, None, :]  # (B, S, S)
        
        # kv index is among q's foreign -> primary neighbors
        kv_in_f2p = (node_idxs[:, None, :, None] == f2p_nbr_idxs[:, :, None, :]).any(-1)  # (B, S, S)
        
        # q index is among kv's primary -> foreign neighbors (reverse relation)
        q_in_f2p = (node_idxs[:, :, None, None] == f2p_nbr_idxs[:, None, :, :]).any(-1)  # (B, S, S)
        
        # Same column AND same table
        same_col_table = (col_name_idxs[:, :, None] == col_name_idxs[:, None, :]) & (
            table_name_idxs[:, :, None] == table_name_idxs[:, None, :]
        )  # (B, S, S)
        
        # Final boolean masks
        attn_masks = {
            "feat": (same_node | kv_in_f2p) & pad,
            "nbr": q_in_f2p & pad,
            "col": same_col_table & pad,
            "full": pad,
        }
        
        # Make contiguous for better performance
        for l in attn_masks:
            attn_masks[l] = attn_masks[l].contiguous()
        
        # Convert to block masks
        from functools import partial
        make_block_mask = partial(
            _make_block_mask,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
        )
        block_masks = {
            l: make_block_mask(attn_mask) for l, attn_mask in attn_masks.items()
        }
        
        # Encode cells (same as RT)
        x = 0
        x = x + (
            self.rt_model.norm_dict["col_name"](
                self.rt_model.enc_dict["col_name"](batch["col_name_values"])
            )
            * (~is_padding)[..., None]
        )
        
        for i, t in enumerate(["number", "text", "datetime", "boolean"]):
            x = x + (
                self.rt_model.norm_dict[t](self.rt_model.enc_dict[t](batch[t + "_values"]))
                * ((batch["sem_types"] == i) & ~batch["masks"] & ~is_padding)[..., None]
            )
            x = x + (
                self.rt_model.mask_embs[t]
                * ((batch["sem_types"] == i) & batch["masks"] & ~is_padding)[..., None]
            )
        
        # Relational blocks
        for block in self.rt_model.blocks:
            x = block(x, block_masks)
        
        # Final norm (this is what we return - embeddings)
        x = self.rt_model.norm_out(x)
        
        return x  # (B, S, d_model)


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
        """Get text embedding for a cell value."""
        if self.text_embedder is not None:
            try:
                emb = self.text_embedder([str(text_value)])
                if isinstance(emb, np.ndarray):
                    return emb[0] if emb.ndim > 1 else emb
                return emb[0] if isinstance(emb, list) else emb
            except:
                pass
        # Return zero embedding as fallback
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
        device = batch[list(batch.node_types)[0]].seed_time.device
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
        batch_size: int
    ) -> Dict[str, Tensor]:
        """
        Convert HeteroData batch to RT batch format.
        
        Args:
            batch: HeteroData from NeighborLoader
            entity_table: Target entity table name
            batch_size: Number of target entities
            
        Returns:
            rt_batch: Dict with RT-required fields
        """
        self._build_mappings(batch)
        device = batch[entity_table].seed_time.device
        
        # Extract cells from TensorFrame
        all_cells = []
        node_idxs_list = []
        col_name_idxs_list = []
        table_name_idxs_list = []
        sem_types_list = []
        
        cell_idx = 0
        node_idx_offset = {}  # Track node index offsets per table
        
        for node_type in batch.node_types:
            tf = batch.tf_dict[node_type]
            num_nodes = len(tf)
            
            # Track node index offset for this table
            node_idx_offset[node_type] = cell_idx
            
            # Extract cells from TensorFrame
            for node_idx_in_table in range(num_nodes):
                # Map to global node index
                global_node_idx = cell_idx
                
                for stype_name, feat in tf.feat_dict.items():
                    col_names = tf.col_names_dict.get(stype_name, [])
                    
                    if stype_name == stype.numerical:
                        # Numerical columns
                        if isinstance(feat, Tensor):
                            for col_idx, col_name in enumerate(col_names):
                                value = feat[node_idx_in_table, col_idx].item()
                                all_cells.append({
                                    'value': value,
                                    'sem_type': 0,  # number
                                    'node_idx': global_node_idx,
                                    'col_name': col_name,
                                    'table_name': node_type,
                                })
                                node_idxs_list.append(global_node_idx)
                                col_name_idxs_list.append(self._get_col_idx(col_name, node_type))
                                table_name_idxs_list.append(self._get_table_idx(node_type))
                                sem_types_list.append(0)
                    
                    elif stype_name == stype.categorical:
                        # Categorical columns (treat as number - category index)
                        if isinstance(feat, Tensor):
                            for col_idx, col_name in enumerate(col_names):
                                value = feat[node_idx_in_table, col_idx].item()
                                all_cells.append({
                                    'value': float(value),
                                    'sem_type': 0,  # number
                                    'node_idx': global_node_idx,
                                    'col_name': col_name,
                                    'table_name': node_type,
                                })
                                node_idxs_list.append(global_node_idx)
                                col_name_idxs_list.append(self._get_col_idx(col_name, node_type))
                                table_name_idxs_list.append(self._get_table_idx(node_type))
                                sem_types_list.append(0)
                    
                    elif stype_name == stype.text_embedded:
                        # Text embedded columns
                        if isinstance(feat, torch_frame.data.MultiEmbeddingTensor):
                            for col_idx, col_name in enumerate(col_names):
                                offset = feat.offset
                                start_idx = offset[node_idx_in_table * len(col_names) + col_idx]
                                end_idx = offset[node_idx_in_table * len(col_names) + col_idx + 1]
                                value = feat.values[start_idx:end_idx].cpu().numpy()
                                
                                all_cells.append({
                                    'value': value,
                                    'sem_type': 1,  # text
                                    'node_idx': global_node_idx,
                                    'col_name': col_name,
                                    'table_name': node_type,
                                })
                                node_idxs_list.append(global_node_idx)
                                col_name_idxs_list.append(self._get_col_idx(col_name, node_type))
                                table_name_idxs_list.append(self._get_table_idx(node_type))
                                sem_types_list.append(1)
                    
                    elif stype_name == stype.timestamp:
                        # Timestamp columns (treat as datetime)
                        if isinstance(feat, Tensor):
                            for col_idx, col_name in enumerate(col_names):
                                value = feat[node_idx_in_table, col_idx].item()
                                all_cells.append({
                                    'value': value,
                                    'sem_type': 2,  # datetime
                                    'node_idx': global_node_idx,
                                    'col_name': col_name,
                                    'table_name': node_type,
                                })
                                node_idxs_list.append(global_node_idx)
                                col_name_idxs_list.append(self._get_col_idx(col_name, node_type))
                                table_name_idxs_list.append(self._get_table_idx(node_type))
                                sem_types_list.append(2)
                
                cell_idx += 1
        
        # Convert to RT batch format
        seq_len = len(all_cells)
        
        if seq_len == 0:
            # Return empty batch
            return {
                'node_idxs': torch.zeros(1, 0, dtype=torch.long, device=device),
                'col_name_idxs': torch.zeros(1, 0, dtype=torch.long, device=device),
                'table_name_idxs': torch.zeros(1, 0, dtype=torch.long, device=device),
                'sem_types': torch.zeros(1, 0, dtype=torch.long, device=device),
                'is_padding': torch.ones(1, 0, dtype=torch.bool, device=device),
                'masks': torch.zeros(1, 0, dtype=torch.bool, device=device),
                'f2p_nbr_idxs': torch.full((1, 0, 5), -1, dtype=torch.long, device=device),
                'number_values': torch.zeros(1, 0, 1, device=device),
                'text_values': torch.zeros(1, 0, self.d_text, device=device),
                'datetime_values': torch.zeros(1, 0, 1, device=device),
                'boolean_values': torch.zeros(1, 0, 1, device=device),
                'col_name_values': torch.zeros(1, 0, self.d_text, device=device),
            }
        
        # Build RT batch dict
        rt_batch = {
            'node_idxs': torch.tensor(node_idxs_list, device=device).unsqueeze(0),  # (1, S)
            'col_name_idxs': torch.tensor(col_name_idxs_list, device=device).unsqueeze(0),
            'table_name_idxs': torch.tensor(table_name_idxs_list, device=device).unsqueeze(0),
            'sem_types': torch.tensor(sem_types_list, device=device).unsqueeze(0),
            'is_padding': torch.zeros(1, seq_len, dtype=torch.bool, device=device),
            'masks': torch.zeros(1, seq_len, dtype=torch.bool, device=device),  # No masking for inference
        }
        
        # Extract cell values by semantic type
        number_values = []
        text_values = []
        datetime_values = []
        boolean_values = []
        col_name_values = []
        
        for cell in all_cells:
            col_name_emb = self._get_text_embedding(cell['col_name'])
            col_name_values.append(col_name_emb)
            
            if cell['sem_type'] == 0:  # number
                number_values.append([cell['value']])
            elif cell['sem_type'] == 1:  # text
                if isinstance(cell['value'], np.ndarray):
                    text_values.append(cell['value'])
                else:
                    text_values.append(self._get_text_embedding(cell['value']))
            elif cell['sem_type'] == 2:  # datetime
                datetime_values.append([cell['value']])
            elif cell['sem_type'] == 3:  # boolean
                boolean_values.append([float(cell['value'] > 0)])
        
        # Convert to tensors
        rt_batch['number_values'] = torch.tensor(
            number_values, device=device, dtype=torch.float32
        ).unsqueeze(0) if number_values else torch.zeros(1, 0, 1, device=device)
        
        rt_batch['text_values'] = torch.tensor(
            text_values, device=device, dtype=torch.float32
        ).unsqueeze(0) if text_values else torch.zeros(1, 0, self.d_text, device=device)
        
        rt_batch['datetime_values'] = torch.tensor(
            datetime_values, device=device, dtype=torch.float32
        ).unsqueeze(0) if datetime_values else torch.zeros(1, 0, 1, device=device)
        
        rt_batch['boolean_values'] = torch.tensor(
            boolean_values, device=device, dtype=torch.float32
        ).unsqueeze(0) if boolean_values else torch.zeros(1, 0, 1, device=device)
        
        rt_batch['col_name_values'] = torch.tensor(
            col_name_values, device=device, dtype=torch.float32
        ).unsqueeze(0) if col_name_values else torch.zeros(1, 0, self.d_text, device=device)
        
        # Compute f2p_nbr_idxs from edge_index_dict
        rt_batch['f2p_nbr_idxs'] = self._compute_f2p_neighbors(
            batch, node_idxs_list, table_name_idxs_list
        )
        
        return rt_batch


def aggregate_cells_to_nodes(
    cell_embeds: Tensor,  # (B, S, d_model)
    rt_batch: Dict[str, Tensor],
    batch: HeteroData,
    entity_table: NodeType,
    aggr: str = "sum"
) -> Dict[NodeType, Tensor]:
    """
    Aggregate RT cell embeddings to node-level embeddings.
    
    Args:
        cell_embeds: RT encoder output (B, S, d_model)
        rt_batch: RT batch dict with node_idxs, table_name_idxs
        batch: Original HeteroData batch
        entity_table: Target entity table
        aggr: Aggregation method ("sum", "mean", "max")
        
    Returns:
        x_dict: Dict[NodeType, Tensor] with (num_nodes, d_model)
    """
    device = cell_embeds.device
    node_idxs = rt_batch['node_idxs'][0]  # (S,)
    table_name_idxs = rt_batch['table_name_idxs'][0]  # (S,)
    cell_embeds = cell_embeds[0]  # (S, d_model)
    
    # Build reverse mapping: table_idx -> table_name
    table_idx_to_name = {}
    for idx, node_type in enumerate(batch.node_types):
        table_idx_to_name[idx] = node_type
    
    x_dict = {}
    
    for node_type in batch.node_types:
        table_idx = list(batch.node_types).index(node_type)
        mask = (table_name_idxs == table_idx)
        
        if not mask.any():
            # No cells for this node type
            x_dict[node_type] = torch.zeros(0, cell_embeds.shape[-1], device=device)
            continue
        
        # Get cells for this table
        table_node_idxs = node_idxs[mask]
        table_cell_embeds = cell_embeds[mask]
        
        # Group cells by node_idx
        node_embeds_dict = {}
        for i, node_idx in enumerate(table_node_idxs):
            node_idx_val = node_idx.item() if isinstance(node_idx, Tensor) else node_idx
            if node_idx_val not in node_embeds_dict:
                node_embeds_dict[node_idx_val] = []
            node_embeds_dict[node_idx_val].append(table_cell_embeds[i])
        
        # Aggregate (match Rel-LLM's aggregation method)
        if node_embeds_dict:
            aggregated = []
            for node_idx in sorted(node_embeds_dict.keys()):
                cells = torch.stack(node_embeds_dict[node_idx])
                if aggr == "sum":
                    aggregated.append(cells.sum(dim=0))
                elif aggr == "mean":
                    aggregated.append(cells.mean(dim=0))
                elif aggr == "max":
                    aggregated.append(cells.max(dim=0)[0])
                else:
                    raise ValueError(f"Unknown aggregation method: {aggr}. Must be 'sum', 'mean', or 'max'.")
            
            if aggregated:
                x_dict[node_type] = torch.stack(aggregated)
            else:
                x_dict[node_type] = torch.zeros(0, cell_embeds.shape[-1], device=device)
        else:
            x_dict[node_type] = torch.zeros(0, cell_embeds.shape[-1], device=device)
    
    return x_dict

