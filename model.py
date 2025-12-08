from typing import Any, Dict, List, Optional
import contextlib
import random
import copy

import torch
from torch import Tensor
from torch.nn import Embedding, ModuleDict, Sigmoid, Sequential, Linear, Dropout

import torch_frame.data
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType
from transformers import AutoModelForCausalLM, AutoTokenizer

from relbench.base import TaskType
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder
from torch_frame.utils.infer_stype import infer_series_stype
from torch_frame import stype
from utils import question_dict, description_dict, initialize_weights

# RT integration
try:
    from rt_adapter import RTEncoderOnly, HeteroDataToRTBatch, aggregate_cells_to_nodes
    RT_AVAILABLE = True
except ImportError as e:
    RT_AVAILABLE = False
    print(f"Warning: RT adapter not available. RT integration disabled.")
    print(f"Import error: {e}")

# llama model type: https://huggingface.co/meta-llama
# encode special tokens for Llama 3.2: https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-3/
BOS = '<|begin_of_text|>'
EOS_USER = '<|eot_id|>'  # end of the message in a turn
EOS = '<|end_of_text|>'
IGNORE_INDEX = -100  # default = -100 in Pytorch CrossEntropyLoss, https://github.com/huggingface/transformers/issues/29819
accept_stypes = [stype.numerical, stype.categorical, stype.text_tokenized, stype.multicategorical, stype.text_embedded]   # no timestamp


class Model(torch.nn.Module):

    def __init__(self, data: HeteroData, col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]], num_layers: int, channels: int, out_channels: int, aggr: str,
                 norm: str = "batch_norm", dropout=0.0, shallow_list: List[NodeType] = [],  # List of node types to add shallow embeddings to input
                 id_awareness: bool = False, model_type: str = "meta-llama/Llama-3.2-1B", max_new_tokens=1, llm_frozen=False, output_mlp=False, output_probs=True, num_demo=4,
                 dataset=None, task=None, gamma=2.0, alpha=[1.0, 1.0], mask_ratio=0.5, pretrain_random_table=False, pretrain_mask_cell=True,
                 use_rt_encoder: bool = False, rt_config: Optional[Dict] = None, text_embedder=None):
        super().__init__()
        self.aggr = aggr  # Store aggregation method for RT cell-to-node aggregation
        self.use_rt_encoder = use_rt_encoder and RT_AVAILABLE
        
        if self.use_rt_encoder:
            # RT encoder path
            if rt_config is None:
                rt_config = {}
            
            # Import RT model
            import sys
            from pathlib import Path
            rt_path = Path(__file__).parent / "relational-transformer"
            if rt_path.exists():
                sys.path.insert(0, str(rt_path))
                from rt.model import RelationalTransformer
                
                # Initialize RT model
                self.rt_model = RelationalTransformer(
                    num_blocks=rt_config.get('num_blocks', 4),
                    d_model=rt_config.get('d_model', 512),
                    d_text=rt_config.get('d_text', 384),
                    num_heads=rt_config.get('num_heads', 8),
                    d_ff=rt_config.get('d_ff', 2048),
                )
                
                # Wrap for encoder-only output
                self.rt_encoder = RTEncoderOnly(self.rt_model)
                
                # NOTE: HeteroDataToRTBatch is kept for reference but NOT USED when RT sampler is enabled
                # RT sampler returns RT batch format directly - no conversion needed
                # This adapter is only kept for backward compatibility (not used in RT sampler path)
                self.rt_adapter = HeteroDataToRTBatch(
                    col_stats_dict=col_stats_dict,
                    text_embedder=text_embedder,
                    d_text=rt_config.get('d_text', 384)
                )
                
                # Projection from RT d_model to Rel-LLM channels
                rt_d_model = rt_config.get('d_model', 512)
                self.rt_to_channels = Linear(rt_d_model, channels)
                
                # Load pretrained weights if available
                if rt_config.get('pretrained_path'):
                    try:
                        state_dict = torch.load(rt_config['pretrained_path'], map_location='cpu')
                        self.rt_model.load_state_dict(state_dict, strict=False)
                        print(f"Loaded RT pretrained weights from {rt_config['pretrained_path']}")
                    except Exception as e:
                        print(f"Warning: Could not load RT pretrained weights: {e}")
                
                # Convert RT model to bfloat16 (matching RT's behavior: net = net.to(torch.bfloat16))
                # RT's dataset outputs bfloat16 values, so model must also be bfloat16
                self.rt_model = self.rt_model.to(torch.bfloat16)
                # Update the wrapper to use the converted model
                self.rt_encoder = RTEncoderOnly(self.rt_model)
                
                print(f"Using RT encoder: d_model={rt_d_model}, num_blocks={rt_config.get('num_blocks', 4)}")
            else:
                print("Warning: relational-transformer not found. Falling back to GNN.")
                self.use_rt_encoder = False
        
        if not self.use_rt_encoder:
            # Original HeteroEncoder + GNN path
            self.encoder = HeteroEncoder(channels=channels, node_to_col_names_dict={node_type: data[node_type].tf.col_names_dict for node_type in data.node_types},
                                     node_to_col_stats=col_stats_dict, )
            self.gnn = HeteroGraphSAGE(node_types=data.node_types, edge_types=data.edge_types, channels=channels, aggr=aggr, num_layers=num_layers)
        
        self.temporal_encoder = HeteroTemporalEncoder(node_types=[node_type for node_type in data.node_types if "time" in data[node_type]], channels=channels, )
        self.head = MLP(channels, out_channels=out_channels, norm=norm, num_layers=1, dropout=dropout)
        self.embedding_dict = ModuleDict({node: Embedding(data.num_nodes_dict[node], channels) for node in shallow_list})
        self.id_awareness_emb = Embedding(1, channels) if id_awareness else None
        self.output_mlp = output_mlp
        self.output_probs = output_probs
        self.gamma = gamma
        self.alpha = alpha

        # pretrain setup
        self.pretrain_mask_cell = pretrain_mask_cell
        self.pretrain_random_table = pretrain_random_table
        self.mask_ratio = mask_ratio
        self.mask_embed = Embedding(1, channels)
        self.column_keep = {}

        # https://huggingface.co/meta-llama/Llama-3.2-1B
        if model_type == 'gnn':
            self.model = None
            print('Using default GNNs without LLMs')
        else:
            print('Loading LLAMA')
            self.num_demo = num_demo
            self.dataset = dataset
            self.task = task
            self.max_new_tokens = max_new_tokens  # only 1 number for classification but can be multiple for regression  # TODO: how many is the optimal?
            self.tokenizer = AutoTokenizer.from_pretrained(model_type, use_fast=False,
                                                           padding_side="left")  # https://huggingface.co/docs/transformers/llm_tutorial#wrong-padding-side
            self.tokenizer.pad_token = self.tokenizer.eos_token  # for padding, https://huggingface.co/meta-llama/Meta-Llama-3-8B/discussions/36
            self.tokenizer.add_special_tokens({'mask_token': '<MASK>'})  # add masked token
            model = AutoModelForCausalLM.from_pretrained(model_type, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map={"": 0})  # 16 instead of 32 with less memory!
            model.resize_token_embeddings(len(self.tokenizer))  # expand vocab due to '<MASK>', https://huggingface.co/docs/transformers/en/main_classes/tokenizer
            if llm_frozen:
                print("Freezing LLAMA!")
                for name, param in model.named_parameters():
                    param.requires_grad = False
            else:
                print("Training LLAMA with LORA!")  # TODO: use_dora=True
                from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
                model = prepare_model_for_kbit_training(model)
                lora_r: int = 8
                lora_alpha: int = 16
                lora_dropout: float = 0.05
                lora_target_modules = ["q_proj", "v_proj", ]
                config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=lora_target_modules, lora_dropout=lora_dropout, bias="none", task_type="CAUSAL_LM")
                model = get_peft_model(model, config)

            self.model = model
            # Make LLaMA cheaper / more memory-friendly
            self.model.config.use_cache = False          # no KV cache during training
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            # Use bf16 on GPU if available (saves memory vs fp16/fp32)
            if torch.cuda.is_available():
                self.model.to(dtype=torch.bfloat16)
            self.word_embedding = self.model.model.get_input_embeddings()
            if model_type == "Qwen/Qwen2.5-7B-Instruct":
                out_dim = 3584
            elif model_type == "meta-llama/Llama-3.2-1B":
                out_dim = 2048
            self.projector = Sequential(Linear(channels, 1024), Sigmoid(), Dropout(dropout), Linear(1024, out_dim), Dropout(dropout)).to(self.model.device)
            self.lm_head = MLP(out_dim, out_channels=out_channels, norm=norm, num_layers=1, dropout=dropout) if self.output_mlp else None

            # cached token embeddings
            self.bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
            self.pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.model.device)).unsqueeze(0)

        self.reset_parameters()

    def reset_parameters(self):
        if not self.use_rt_encoder:
            self.encoder.reset_parameters()
            self.gnn.reset_parameters()
        else:
            # RT model parameters are initialized in RelationalTransformer.__init__
            if hasattr(self, 'rt_to_channels'):
                torch.nn.init.kaiming_uniform_(self.rt_to_channels.weight, nonlinearity='linear')
                if self.rt_to_channels.bias is not None:
                    torch.nn.init.zeros_(self.rt_to_channels.bias)
        self.temporal_encoder.reset_parameters()
        self.head.reset_parameters()
        for embedding in self.embedding_dict.values():
            torch.nn.init.normal_(embedding.weight, std=0.1)
        if self.id_awareness_emb is not None:
            self.id_awareness_emb.reset_parameters()
        if self.model is not None:
            self.projector.apply(initialize_weights)
            if self.lm_head is not None:
                self.lm_head.reset_parameters()

    @property
    def device(self):
        return list(self.parameters())[0].device

    @property
    def eos_user_id_list(self):
        return self.tokenizer(EOS_USER, add_special_tokens=False).input_ids

    @property
    def eos_id_list(self):
        return self.tokenizer(EOS, add_special_tokens=False).input_ids  # LLAMA tokenizer does not add an eos_token_id at the end of inputs

    @property
    def false_id(self):
        return self.tokenizer('No', add_special_tokens=False).input_ids[0]

    @property
    def true_id(self):
        return self.tokenizer('Yes', add_special_tokens=False).input_ids[0]

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast; if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        return contextlib.nullcontext()

    def encode(self, batch, entity_table, split=None):
        """
        Encode batch to node embeddings.
        
        Args:
            batch: Either HeteroData (from NeighborLoader) or RT batch dict (from RT sampler)
            entity_table: Entity table name
            split: Split name ("train", "val", "test") - needed for RT sampler
        """
        # Check if batch is RT format (dict) or HeteroData
        is_rt_batch = isinstance(batch, dict) and 'node_idxs' in batch

        if is_rt_batch:
            # RT batch from DataLoader (CPU tensors)
            rt_batch_cpu = batch

            if not hasattr(self, 'rt_bridge_dict') or split is None:
                raise ValueError("RT batch format requires rt_bridge_dict and split")
            bridge = self.rt_bridge_dict[split]
            
            # Clear previous seed node IDs to avoid stale data
            self._rt_seed_node_ids = None
            self._rt_seed_node_ids_per_batch = None

            # ---------- 1) metadata on CPU ----------
            if not hasattr(self, '_encode_call_count'):
                self._encode_call_count = 0
            if self._encode_call_count == 0:
                print("[DEBUG] Building metadata from RT batch...", flush=True)

            metadata = bridge.build_metadata(rt_batch_cpu, split)

            if self._encode_call_count == 0:
                print(f"[DEBUG] Metadata built, batch_size={metadata['batch_size']}", flush=True)

            seed_time = metadata['seed_time']
            n_id_dict = metadata['n_id']
            time_dict = metadata['time_dict']
            batch_dict = metadata['batch_dict']
            
            # Store n_id_dict for seed node mapping in forward()
            self._rt_n_id_dict = n_id_dict
            
            # IMPORTANT: Use true_batch_size from RT sampler (number of seed examples)
            # This is the actual batch size, not the number of unique nodes in subgraph
            true_bs = rt_batch_cpu.get("true_batch_size", None)
            if true_bs is not None:
                batch_size = int(true_bs)
                if self._encode_call_count == 0:
                    print(f"[DEBUG] Overriding batch_size with true_batch_size={batch_size}", flush=True)
            else:
                # Fallback: use seed_time length
                batch_size = seed_time.shape[0]
                if self._encode_call_count == 0:
                    print(f"[DEBUG] No true_batch_size found, using seed_time length={batch_size}", flush=True)
            
            # Keep seed_time length consistent with batch_size
            seed_time = seed_time[:batch_size]

            # ---------- 1.5) Extract seed node IDs and map to positions in x_dict ----------
            # For precise seed node selection, we need to map seed node IDs to their positions in aggregated x_dict
            # Extract seed node IDs from RT batch (is_task_nodes marks seed cells)
            is_task_nodes = rt_batch_cpu['is_task_nodes']  # (B, S)
            node_idxs = rt_batch_cpu['node_idxs']  # (B, S) - global node indices
            table_name_idxs = rt_batch_cpu['table_name_idxs']  # (B, S)
            
            # Get table index for entity_table
            entity_table_idx = bridge.node_type_to_table_idx.get(entity_table, None)
            
            seed_node_ids = []
            for b in range(min(batch_size, is_task_nodes.shape[0])):
                batch_is_task = is_task_nodes[b]  # (S,)
                batch_node_idxs = node_idxs[b]  # (S,)
                batch_table_idxs = table_name_idxs[b]  # (S,)
                
                # Find seed cells that belong to entity_table
                seed_mask = batch_is_task & (batch_table_idxs == entity_table_idx)
                seed_node_ids_batch = batch_node_idxs[seed_mask].unique()
                seed_node_ids.append(seed_node_ids_batch)
            
            # Flatten and get unique seed node IDs
            if seed_node_ids:
                all_seed_node_ids = torch.cat(seed_node_ids).unique()
            else:
                all_seed_node_ids = torch.tensor([], dtype=torch.long)
            
            # Store seed node IDs for later use in forward()
            self._rt_seed_node_ids = all_seed_node_ids
            
            # Store seed node IDs per batch item (in order) for label extraction
            # This preserves the batch order needed for label_tokenize
            self._rt_seed_node_ids_per_batch = []
            for b in range(min(batch_size, is_task_nodes.shape[0])):
                batch_is_task = is_task_nodes[b]  # (S,)
                batch_node_idxs = node_idxs[b]  # (S,)
                batch_table_idxs = table_name_idxs[b]  # (S,)
                
                # Find first seed cell that belongs to entity_table (one per batch item)
                seed_mask = batch_is_task & (batch_table_idxs == entity_table_idx)
                if seed_mask.any():
                    seed_node_id = batch_node_idxs[seed_mask][0].item()  # Take first seed node
                    self._rt_seed_node_ids_per_batch.append(seed_node_id)
                else:
                    # Fallback: shouldn't happen, but handle gracefully
                    self._rt_seed_node_ids_per_batch.append(None)

            # ---------- 2) move RT batch to device for RT encoder ----------
            if self._encode_call_count == 0:
                print("[DEBUG] Moving RT batch to device for RT encoder...", flush=True)

            rt_batch_gpu = {
                k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                for k, v in rt_batch_cpu.items()
            }

            if self._encode_call_count == 0:
                print("[DEBUG] Calling RT encoder forward...", flush=True)

            cell_embeds = self.rt_encoder(rt_batch_gpu)  # (B, S, d_model)

            if self._encode_call_count == 0:
                print(f"[DEBUG] RT encoder forward complete, cell_embeds shape={cell_embeds.shape}", flush=True)

            # ---------- 3) aggregate cells → nodes ----------
            from rt_adapter import aggregate_cells_to_nodes_from_rt_batch
            if self._encode_call_count == 0:
                print("[DEBUG] Aggregating cells to nodes...", flush=True)

            x_dict = aggregate_cells_to_nodes_from_rt_batch(
                cell_embeds,
                rt_batch_cpu,                   # CPU batch contains table_ids / mapping
                bridge.table_idx_to_node_type,
                aggr=self.aggr,
            )

            if self._encode_call_count == 0:
                print(f"[DEBUG] Aggregation complete, x_dict keys={list(x_dict.keys())}", flush=True)

            # ---------- 4a) Remap RT task table names to Rel-LLM entity table names ----------
            # RT uses task table names (e.g., "user-churn") but Rel-LLM expects entity table names (e.g., "customer")
            # Map the RT task table name to the entity_table
            remapped_x_dict = {}
            if self._encode_call_count == 0:
                print(f"[DEBUG] Remapping: rt_task_table_name={bridge.rt_task_table_name}, entity_table={entity_table}", flush=True)
                print(f"[DEBUG] Available RT tables in batch: {list(x_dict.keys())}", flush=True)
            
            # Check if the expected task table is in the batch
            if bridge.rt_task_table_name in x_dict:
                # Map the task table to entity_table
                remapped_x_dict[entity_table] = x_dict[bridge.rt_task_table_name]
                if self._encode_call_count == 0:
                    print(f"[DEBUG] Mapped {bridge.rt_task_table_name} -> {entity_table}", flush=True)
            else:
                # The expected task table is not in x_dict - this shouldn't happen but handle gracefully
                # Try to find which table corresponds to entity_table by checking table_info
                # For now, if we only have one table, assume it's the entity table
                if len(x_dict) == 1:
                    rt_table_name = list(x_dict.keys())[0]
                    remapped_x_dict[entity_table] = x_dict[rt_table_name]
                    if self._encode_call_count == 0:
                        print(f"[DEBUG] WARNING: Expected {bridge.rt_task_table_name} but got {rt_table_name}, mapping to {entity_table}", flush=True)
                else:
                    # Multiple tables - need to figure out which one is the entity table
                    # Check if any RT table name matches patterns that suggest it's the entity table
                    # For user-churn task, entity_table is "customer", RT might use "user-churn" or similar
                    found = False
                    for rt_table_name, emb in x_dict.items():
                        # Check if this RT table corresponds to our task
                        if rt_table_name == bridge.rt_task_table_name or rt_table_name.startswith(bridge.rt_task_table_name.split('-')[0]):
                            remapped_x_dict[entity_table] = emb
                            found = True
                            if self._encode_call_count == 0:
                                print(f"[DEBUG] Mapped {rt_table_name} -> {entity_table} (pattern match)", flush=True)
                            break
                    
                    if not found:
                        # Last resort: use the first table and warn
                        rt_table_name = list(x_dict.keys())[0]
                        remapped_x_dict[entity_table] = x_dict[rt_table_name]
                        if self._encode_call_count == 0:
                            print(f"[DEBUG] WARNING: Could not find matching table, using first table {rt_table_name} -> {entity_table}", flush=True)
            
            # Keep other tables as-is (they might be other task tables or DB tables)
            for rt_table_name, emb in x_dict.items():
                if rt_table_name != bridge.rt_task_table_name:
                    # Try to map if we have a mapping, otherwise keep RT name
                    rel_llm_name = bridge.rt_to_relllm_table_map.get(rt_table_name, rt_table_name)
                    if rel_llm_name != entity_table:  # Don't overwrite entity_table
                        remapped_x_dict[rel_llm_name] = emb
            
            x_dict = remapped_x_dict

            if self._encode_call_count == 0:
                print(f"[DEBUG] After remapping, x_dict keys={list(x_dict.keys())}", flush=True)

            self._encode_call_count += 1

            # ---------- 4b) project RT d_model → channels ----------
            x_dict = {node_type: self.rt_to_channels(emb) for node_type, emb in x_dict.items()}

        else:
            # HeteroData format (from NeighborLoader)
            # If RT encoder is enabled, we should NEVER get HeteroData - RT sampler must be used
            if self.use_rt_encoder:
                raise ValueError(
                    "RT encoder is enabled but received HeteroData batch. "
                    "RT encoder REQUIRES RT sampler (not NeighborLoader). "
                    "When --use_rt_encoder is set, RT's sampler is used automatically. "
                    "HeteroDataToRTBatch conversion is NOT used - RT sampler returns RT batch format directly."
                )
            
            # Original Rel-LLM path (GNN only, no RT encoder)
            seed_time = batch[entity_table].seed_time
            batch_size = len(seed_time)
            x_dict = self.encoder(batch.tf_dict)  # HeteroEncoder
            
            # Use HeteroData metadata
            n_id_dict = {node_type: batch[node_type].n_id for node_type in batch.node_types}
            time_dict = batch.time_dict
            batch_dict = batch.batch_dict

        # Apply temporal encoding
        rel_time_dict = self.temporal_encoder(seed_time, time_dict, batch_dict)
        for node_type, rel_time in rel_time_dict.items():
            if node_type in x_dict:
                x_dict[node_type] = x_dict[node_type] + rel_time
        
        # Apply ID embeddings
        for node_type, embedding in self.embedding_dict.items():
            if node_type in x_dict and node_type in n_id_dict:
                x_dict[node_type] = x_dict[node_type] + embedding(n_id_dict[node_type])
        
        return x_dict, batch_size

    def column_filter(self, df, df_name):
        if df_name not in self.column_keep:
            self.column_keep[df_name] = [col for col in df.columns if infer_series_stype(df[col]) in accept_stypes]
        return self.column_keep[df_name]

    def pretrain(self, batch, entity_table, split: Optional[str] = None):
        # Handle RT batch format vs HeteroData
        is_rt_batch = isinstance(batch, dict) and 'node_idxs' in batch
        
        # If RT encoder is enabled, we should NEVER get HeteroData
        if self.use_rt_encoder and not is_rt_batch:
            raise ValueError(
                "RT encoder is enabled but received HeteroData batch in pretrain(). "
                "RT encoder requires RT sampler (not NeighborLoader)."
            )
        
        if is_rt_batch:
            # RT batch - extract batch_size from metadata
            if not hasattr(self, 'rt_bridge_dict') or split is None:
                raise ValueError("RT batch requires rt_bridge_dict and split")
            bridge = self.rt_bridge_dict[split]
            _, batch_size = bridge.extract_seed_nodes(batch)
            select_table = entity_table
            # RT batches don't support pretraining masking yet - would need to implement
            raise NotImplementedError("Pretraining with RT sampler not yet implemented")
        else:
            # Original HeteroData path (GNN only)
            select_table = entity_table
        batch_size = len(batch[entity_table].seed_time)
        num_tokens_to_mask = int(batch_size * self.mask_ratio)  # Number of tokens to mask
        mask_indices = torch.randperm(batch_size)[:num_tokens_to_mask].to(self.device)
        if self.pretrain_mask_cell:
            select_column = random.choice([k for k, v in batch[entity_table].tf._col_to_stype_idx.items() if v[0] != stype.timestamp])  # exclude timestamp
            select_stype, select_idx = batch[entity_table].tf._col_to_stype_idx[select_column]
            select_feat = batch[entity_table].tf.feat_dict[select_stype]
            if isinstance(select_feat, torch_frame.data.MultiEmbeddingTensor):    # MultiEmbeddingTensor not support value setting...
                mask_values = select_feat.values.clone()
                offset = select_feat.offset
                mask_values[mask_indices, offset[select_idx]: offset[select_idx + 1]] = torch.zeros_like(mask_values[mask_indices, offset[select_idx]: offset[select_idx + 1]])
                batch[entity_table].tf.feat_dict[select_stype].values = mask_values
            elif isinstance(select_feat, torch.Tensor):
                batch[entity_table].tf.feat_dict[select_stype][mask_indices] = torch.zeros_like(select_feat[mask_indices])  # timestamp cannot be masked with 0 (min_year)
            x_dict, _ = self.encode(batch, entity_table)
        else:
            x_dict, _ = self.encode(batch, entity_table)
            if self.pretrain_random_table:
                select_table = random.choice([i for i in x_dict.keys() if x_dict[i].numel() > 0])  # random select a node type
            x_dict[select_table][mask_indices] = self.mask_embed.weight   # mask token embeddings

        if not self.use_rt_encoder:
            x_dict = self.gnn(x_dict, batch.edge_index_dict)  # interactions among different tables
        node_embed = x_dict[select_table][:batch_size]
        node_embed = self.projector(node_embed)

        # Seed entity information
        seed_df_indices = batch[select_table].n_id[mask_indices].cpu().numpy()  # input_id -> the ID of the training table
        seed_df = batch[select_table].df.iloc[seed_df_indices]
        filtered_df = seed_df[self.column_filter(seed_df, select_table)]
        if self.pretrain_mask_cell: filtered_df = filtered_df[[select_column]]
        # print(filtered_df)
        # for col in seed_df.columns:   # TODO: check stype for other datasets
        #     print(infer_series_stype(seed_df[col]), ' : ', seed_df[col].iloc[0])

        batch_input_ids, batch_label_input_ids = [], []
        for index, row in filtered_df.iterrows():  # iterate each sample in the batch
            input_ids, label_input_ids = [], []
            row_dict = list(row.to_dict().items())
            random.shuffle(row_dict)
            for col_name, col_value in row_dict:  # todo: multiple columns?
                if col_value in ['\\N'] and not self.pretrain_mask_cell: continue  # filter no meaningful words
                other_values = [val for val in filtered_df[col_name].dropna().unique() if val != col_value and val != '\\N']
                if random.random() > 0.5 or len(other_values) < 1:
                    prompt = f'{col_name} is {col_value}.'
                    label_tokens = self.true_id
                else:
                    new_value = random.choice(other_values)
                    prompt = f'{col_name} is {new_value}.'
                    label_tokens = self.false_id

                input_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids + self.eos_user_id_list
                label_input_ids += len(input_ids) * [IGNORE_INDEX] + [label_tokens] + self.eos_id_list
                break

            batch_input_ids.append(input_ids)
            batch_label_input_ids.append(label_input_ids)

        # tokenizer happens on CPU
        question = ' Question: Is the statement correct? Give Yes or No as answer.'
        question_embeds = self.word_embedding(self.tokenizer(question, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.device))

        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(num_tokens_to_mask):
            # Add bos & eos token: https://github.com/XiaoxinHe/G-Retriever/issues/17
            # print(self.tokenizer.decode(batch_input_ids[i]))
            inputs_embeds = self.word_embedding(torch.tensor(batch_input_ids[i]).to(self.device))
            inputs_embeds = torch.cat([self.bos_embeds, node_embed[mask_indices[i]].unsqueeze(0), question_embeds, inputs_embeds], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(num_tokens_to_mask):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([self.pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * (max_length - len(batch_label_input_ids[i])) + batch_label_input_ids[i]  # `inputs_embeds` contain `labels`

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.device)

        inputs_embeds = inputs_embeds.to(
            device=self.model.device, dtype=self.model.dtype
        )
        attention_mask = attention_mask.to(device=self.model.device)
        label_input_ids = label_input_ids.to(device=self.model.device)

        with self.maybe_autocast():
            outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True, labels=label_input_ids)
        return outputs.loss

    def label_tokenize(self, batch, entity_table, split=None):
        """
        Tokenize labels for label prompting.

        HeteroData batches:
            - use batch[entity_table].y (original Rel-LLM behavior)
        RT batches:
            - extract labels directly from RT batch using is_targets mask
              and *_values tensors, without going through entity IDs.
        """
        is_rt_batch = isinstance(batch, dict) and 'node_idxs' in batch

        if is_rt_batch:
            if 'is_targets' not in batch:
                raise ValueError("RT batch is missing 'is_targets'")

            is_targets = batch['is_targets']          # (B, S)
            B, S = is_targets.shape

            # IMPORTANT: Use true_batch_size to avoid processing padding items
            # RT pads batches to batch_size, but only first true_batch_size items are real
            true_bs = batch.get("true_batch_size", B)
            if true_bs > B:
                # Sanity check: true_batch_size shouldn't exceed batch dimension
                true_bs = B

            label_values: List[float] = []

            # Only process real batch items (not padding)
            for b in range(true_bs):
                mask = is_targets[b]                  # (S,)
                if not mask.any():
                    raise ValueError(f"No target cell found for RT batch item {b}")

                if self.task.task_type == TaskType.BINARY_CLASSIFICATION:
                    if 'boolean_values' not in batch:
                        raise ValueError("RT batch missing 'boolean_values' for binary task")
                    vals = batch['boolean_values'][b][mask].view(-1)
                    # take first target value
                    v = vals[0].item()
                    # assume >0.5 is True
                    label_values.append(bool(v > 0.5))

                elif self.task.task_type == TaskType.REGRESSION:
                    if 'number_values' not in batch:
                        raise ValueError("RT batch missing 'number_values' for regression task")
                    vals = batch['number_values'][b][mask].view(-1)
                    v = vals[0].item()
                    label_values.append(float(v))

                else:
                    raise ValueError(f"Unsupported task type for RT label_tokenize: {self.task.task_type}")

            # Now convert label_values → strings for tokenizer
            if self.task.task_type == TaskType.BINARY_CLASSIFICATION:
                label_strs = ['Yes' if v else 'No' for v in label_values]
            elif self.task.task_type == TaskType.REGRESSION:
                label_strs = [str(v) for v in label_values]

            labels = self.tokenizer(label_strs, add_special_tokens=False)
            return labels

        # ---------- Original HeteroData path ----------
        if self.task.task_type == TaskType.BINARY_CLASSIFICATION:
            label = ['Yes' if i else 'No' for i in batch[entity_table].y.bool().tolist()]
        elif self.task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            # This branch is effectively unused in your current setup.
            label = [str(i) for i in batch[entity_table].y.int().tolist()]
        elif self.task.task_type == TaskType.REGRESSION:
            label = [str(i) for i in batch[entity_table].y.float().tolist()]
        else:
            raise ValueError(f"Unsupported task type: {self.task.task_type}")

        labels = self.tokenizer(label, add_special_tokens=False)
        return labels
    def get_demo_info(self, demo_batch, entity_table, split=None):
        x_dict, demo_batch_size = self.encode(demo_batch, entity_table, split=split)
        assert self.num_demo <= demo_batch_size, 'Too large demo numbers!'
        if not self.use_rt_encoder:
            x_dict = self.gnn(x_dict, demo_batch.edge_index_dict)
        demo_node_embeds = self.projector(x_dict[entity_table][:demo_batch_size])
        demo_labels = self.label_tokenize(demo_batch, entity_table, split=split).input_ids
        demo_labels = torch.tensor(demo_labels, device=self.device)
        return demo_node_embeds, demo_labels

    def recursive_sample(self, batch_data: HeteroData, node_type: str, target_nodes: torch.Tensor, num_hops: int = 2):
        """
        Recursively samples neighbors from a batch heterogeneous graph while ensuring previously sampled node types are excluded.
        Args:
            batch_data (HeteroData): A batched heterogeneous graph from PyG's NeighborLoader.
            target_nodes (torch.Tensor): The indices of the target nodes in the `node_type`.
            node_type (str): The node type of the target nodes (e.g., "entity_table").
            num_hops (int): Number of recursive hops to sample.
        """
        sampled_nodes = [node_type]  # Track sampled node types to avoid re-sampling
        neighbor_dict = {node_type: {node: {} for node in target_nodes.tolist()}}  # Initialize nested dictionary

        def sample_neighbors(current_nodes, current_node_type, depth, tmp_dict):
            """Recursively sample neighbors up to num_hops while avoiding duplicate node types."""
            if depth == num_hops: return
            next_nodes = {}
            for edge_type in batch_data.edge_types:  # Iterate through edge types to find valid neighbors
                src_type, _, dst_type = edge_type
                if src_type == current_node_type and dst_type not in sampled_nodes:
                    src_nodes = batch_data[edge_type].edge_index[0].tolist()
                    dst_nodes = batch_data[edge_type].edge_index[1].tolist()
                    # print(edge_type, current_node_type, len(src_nodes), len(dst_nodes))
                    for src, dst in zip(src_nodes, dst_nodes):
                        if src in current_nodes:  # Ensure it's a valid node from the current set
                            if dst_type not in tmp_dict[src]:
                                tmp_dict[src][dst_type] = {}
                            tmp_dict[src][dst_type][dst] = {}

                            if dst_type not in next_nodes:
                                next_nodes[dst_type] = set()
                            next_nodes[dst_type].add(dst)
            for node in tmp_dict.keys():
                for next_node_type, nodes in next_nodes.items():   # Recursive call for the next hop
                    if next_node_type in tmp_dict[node].keys():
                        sample_neighbors(nodes, next_node_type, depth + 1, tmp_dict[node][next_node_type])

        sample_neighbors(set(target_nodes.tolist()), node_type, depth=0, tmp_dict=neighbor_dict[node_type])   # Start recursive sampling from target nodes
        return neighbor_dict

    def get_neighbor_embedding(self, neighbor_dict, embed_dict):

        def recursive_collect(node_type, node_id, sub_neighbors):
            """Recursively collect embeddings depth-first for a single node."""
            node_embedding = embed_dict[node_type][node_id].unsqueeze(0)  # Shape: (1, D)
            # Collect embeddings from deeper neighbors recursively
            neighbor_embeds = []
            for sub_type, sub_dict in sub_neighbors.items():
                for sub_id, sub_sub_neighbors in sub_dict.items():
                    neighbor_embeds.append(recursive_collect(sub_type, sub_id, sub_sub_neighbors))
            if neighbor_embeds:
                neighbor_embeds = torch.cat(neighbor_embeds)  # Concatenate along feature dimension
                node_embedding = torch.cat([node_embedding, neighbor_embeds])
            return node_embedding

        all_embeddings = []
        for target_type, targets in neighbor_dict.items():
            for target_id, neighbors in targets.items():
                all_embeddings.append(recursive_collect(target_type, target_id, neighbors))
        return torch.cat(all_embeddings) if all_embeddings else None

    def forward(
        self,
        batch,
        entity_table: NodeType,
        context: bool = True,
        demo_info=None,
        inference: bool = False,
        split: Optional[str] = None,
    ) -> Tensor:
        # 1) Encode graph → node embeddings
        x_dict, batch_size = self.encode(batch, entity_table, split=split)

        # num_sampled_nodes_dict ->  the number of sampled nodes for each node type at each layer (hop)
        # e.g. {'user_friends': [0, 67636, 0], 'users': [512, 0, 2812], ...}
        if not self.use_rt_encoder and not isinstance(batch, dict):
            # GNN only works with HeteroData (has edge_index_dict)
            x_dict = self.gnn(x_dict, batch.edge_index_dict)  # interactions among different tables

        # For RT batches, use precise seed node mapping if available
        if hasattr(self, '_rt_seed_node_ids') and self._rt_seed_node_ids is not None and len(self._rt_seed_node_ids) > 0:
            # Map seed node IDs to their positions in x_dict[entity_table]
            # x_dict[entity_table] corresponds to n_id_dict[entity_table] from metadata
            seed_node_ids = self._rt_seed_node_ids
            if entity_table in x_dict:
                # Get node IDs from metadata (stored in encode)
                if hasattr(self, '_rt_n_id_dict') and entity_table in self._rt_n_id_dict:
                    n_id_entity = self._rt_n_id_dict[entity_table]
                    # Find positions of seed nodes in n_id_entity
                    seed_positions = []
                    for seed_id in seed_node_ids:
                        matches = (n_id_entity == seed_id.item())
                        if matches.any():
                            pos = torch.where(matches)[0][0]
                            seed_positions.append(pos.item())
                    
                    if len(seed_positions) == batch_size:
                        # Use precise seed node positions
                        seed_positions_tensor = torch.tensor(seed_positions, device=x_dict[entity_table].device)
                        node_embed = x_dict[entity_table][seed_positions_tensor]
                    else:
                        # Fallback: use first batch_size nodes
                        if not hasattr(self, '_encode_call_count') or self._encode_call_count == 0:
                            print(f"[DEBUG] WARNING: Found {len(seed_positions)} seed positions but batch_size={batch_size}, using first batch_size nodes", flush=True)
                        node_embed = x_dict[entity_table][:batch_size]
                else:
                    # No metadata available, use first batch_size
                    node_embed = x_dict[entity_table][:batch_size]
            else:
                # Entity table not in x_dict, this shouldn't happen
                raise KeyError(f"Entity table '{entity_table}' not found in x_dict. Available keys: {list(x_dict.keys())}")
        else:
            # Original path: use first batch_size nodes
            node_embed = x_dict[entity_table][:batch_size]
        if self.model is None:
            # Pure GNN mode
            return self.head(node_embed)

        # Project graph embeddings to LLM dimension
        node_embed = self.projector(node_embed)

        # 2) Text side: task description, question, labels
        task_desc = description_dict[self.dataset][self.task.name]
        question = " Question: " + question_dict[self.dataset][self.task.name] + " Answer: "
        task_descs = self.tokenizer(task_desc, add_special_tokens=False)
        questions = self.tokenizer(question, add_special_tokens=False)

        is_rt_batch = isinstance(batch, dict) and 'node_idxs' in batch

        if not inference and not self.output_mlp:
            # Extract labels for label prompting (works for both HeteroData and RT batches)
            labels = self.label_tokenize(batch, entity_table, split=split)

        # graph_prompt = node_embed[i].unsqueeze(0)
        if context and not is_rt_batch:
            neighbors = self.recursive_sample(
                batch, entity_table, torch.arange(batch_size), num_hops=1
            )
        else:
            neighbors = None

        # 3) In-context demos
        if self.num_demo > 0 and demo_info is not None:
            demo_node_embeds, demo_labels = demo_info
            if self.task.task_type == TaskType.BINARY_CLASSIFICATION:
                # Balanced sampling
                mask = demo_labels == demo_labels[0].item()
                indices_A = torch.where(mask)[0]
                indices_B = torch.where(~mask)[0]
                count_A = indices_A.size(0)
                count_B = indices_B.size(0)
                num_demo_half = self.num_demo // 2
                extra = self.num_demo % 2
                assert (
                    count_A >= num_demo_half + extra
                    and count_B >= num_demo_half + extra
                ), "Not enough samples in one class"
                sampled_A = indices_A[
                    torch.randint(0, count_A, (batch_size, num_demo_half), device=self.device)
                ]
                sampled_B = indices_B[
                    torch.randint(0, count_B, (batch_size, num_demo_half), device=self.device)
                ]
                if extra:
                    extra_class = torch.randint(0, 2, (batch_size,), device=self.device)
                    extra_A = indices_A[
                        torch.randint(0, count_A, (batch_size,), device=self.device)
                    ]
                    extra_B = indices_B[
                        torch.randint(0, count_B, (batch_size,), device=self.device)
                    ]
                    extra_samples = torch.where(extra_class, extra_B, extra_A).unsqueeze(1)
                    sampled_indices = torch.cat(
                        [sampled_A, sampled_B, extra_samples], dim=1
                    )
                else:
                    sampled_indices = torch.cat([sampled_A, sampled_B], dim=1)
                shuffle_idx = torch.rand(
                    batch_size, self.num_demo, device=self.device
                ).argsort(dim=1)
                sampled_indices = sampled_indices.gather(1, shuffle_idx)
            else:
                random_matrix = torch.rand(
                    batch_size, len(demo_node_embeds), device=self.device
                )
                sampled_indices = random_matrix.argsort(dim=1)[:, : self.num_demo]

            demo_node_embeds = demo_node_embeds[sampled_indices]  # (B, K, D)
            demo_labels = demo_labels[sampled_indices]            # (B, K, 1)

        # 4) Build per-sample prompts in embedding space
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []

        for i in range(batch_size):
            input_ids = task_descs.input_ids + questions.input_ids + self.eos_user_id_list

            if not inference and not self.output_mlp:
                label_input_ids = labels.input_ids[i] + self.eos_id_list
                input_ids += label_input_ids

            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.device))

            # In-context demos
            if self.num_demo > 0 and demo_info is not None:
                demo_embeds = []
                for k in range(self.num_demo):
                    demo_embeds += [
                        demo_node_embeds[i][k].unsqueeze(0),
                        self.word_embedding(demo_labels[i][k]),
                    ]
                demo_embeds.append(node_embed[i].unsqueeze(0))
                inputs_embeds = torch.cat(
                    [inputs_embeds[:-1], torch.cat(demo_embeds), inputs_embeds[-1:]]
                )

            # Graph prompt: root + neighbors
            graph_prompt = node_embed[i].unsqueeze(0)
            if context and neighbors is not None:
                neighbor_embed = self.get_neighbor_embedding(
                    neighbors[entity_table][i], x_dict
                )
                if neighbor_embed is not None:
                    neighbor_embed = self.projector(neighbor_embed)
                    graph_prompt = torch.cat([graph_prompt, neighbor_embed])

            # BOS + graph prompt + text
            inputs_embeds = torch.cat(
                [self.bos_embeds, graph_prompt, inputs_embeds], dim=0
            )

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

            if not inference and not self.output_mlp:
                # Label positions are only the tail; earlier tokens are IGNORE_INDEX
                label_input_ids = [IGNORE_INDEX] * (
                    inputs_embeds.shape[0] - len(label_input_ids)
                ) + label_input_ids
                batch_label_input_ids.append(label_input_ids)

        # 5) Pad to same length (still on CPU)
        max_length = max(x.shape[0] for x in batch_inputs_embeds)
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            if pad_length > 0:
                batch_inputs_embeds[i] = torch.cat(
                    [self.pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]],
                    dim=0,
                )
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]
            if not inference and not self.output_mlp:
                    batch_label_input_ids[i] = (
                        [IGNORE_INDEX] * pad_length + batch_label_input_ids[i]
                    )

        # Stack on CPU
        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0)
        attention_mask = torch.tensor(batch_attention_mask)
        if not inference and not self.output_mlp:
            label_input_ids = torch.tensor(batch_label_input_ids)

        # Move to LLaMA device / dtype
        inputs_embeds = inputs_embeds.to(
            device=self.model.device, dtype=self.model.dtype
        )
        attention_mask = attention_mask.to(device=self.model.device)
        if not inference and not self.output_mlp:
            label_input_ids = label_input_ids.to(device=self.model.device)

        # 6) Output-MLP mode (no generation, just hidden states → head)
        if self.output_mlp:
            with self.maybe_autocast():
                outputs = self.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    output_hidden_states=True,
                )
            hidden = outputs.hidden_states[-1][..., -1, :]
            # Convert to float32 for lm_head (RT encoder outputs bfloat16, but lm_head expects float32)
            hidden = hidden.to(torch.float32)
            pred = self.lm_head(hidden).view(-1)
            return pred

        # 8) TRAINING BRANCH
        if not inference:
            # Non-binary: use HuggingFace CE loss directly
            if self.task.task_type != TaskType.BINARY_CLASSIFICATION:
                with self.maybe_autocast():
                    outputs = self.model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        return_dict=True,
                        labels=label_input_ids,
                    )
                return outputs.loss

            # Binary: focal loss
            with self.maybe_autocast():
                outputs = self.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                )

            # Shift so tokens < n predict token n
            logits = outputs.logits[..., :-1, :]  # (B, L-1, C)
            labels = label_input_ids[..., 1:]      # (B, L-1)

            valid_mask = labels != IGNORE_INDEX
            if not valid_mask.any():
                # No valid labels; return a dummy grad-bearing zero
                return torch.zeros([], device=self.model.device, requires_grad=True)

            labels_valid = labels[valid_mask]
            logits_valid = logits[valid_mask]

            probs = torch.nn.functional.softmax(logits_valid, dim=-1)
            probs = probs.gather(
                dim=-1, index=labels_valid.unsqueeze(-1)
            ).squeeze(-1)

            focal_weight = (1 - probs).pow(self.gamma)
            loss = -focal_weight * probs.log()

            if self.alpha is not None:
                class_weights = torch.ones(
                    self.model.vocab_size, device=self.model.device
                )
                class_weights[self.false_id] = self.alpha[0]
                class_weights[self.true_id] = self.alpha[1]
                alpha_t = class_weights.gather(dim=0, index=labels_valid)
                loss = alpha_t * loss

            return loss.mean()

        # 9) INFERENCE BRANCH (inference=True): generation + decoding
        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        if self.task.task_type == TaskType.BINARY_CLASSIFICATION:
            if self.output_probs:
                # Use logits of generated token for Yes/No
                scores = outputs.scores[0]
                pred = scores[..., [self.false_id, self.true_id]]
                pred = torch.softmax(pred, dim=-1)[..., 1]
                pred = torch.nan_to_num(pred, nan=0.5)
            else:
                seq = self.tokenizer.batch_decode(
                    outputs.sequences, skip_special_tokens=True
                )
                pred = torch.tensor([0.0 if s == "No" else 1.0 for s in seq])
        elif self.task.task_type == TaskType.REGRESSION:
            seq = self.tokenizer.batch_decode(
                outputs.sequences, skip_special_tokens=True
            )
            vals = []
            for s in seq:
                try:
                    vals.append(float(s))
                except ValueError:
                    vals.append(0.0)
            pred = torch.tensor(vals)
        else:
            # Multiclass classification via decoded token(s) – you can adapt this.
            seq = self.tokenizer.batch_decode(
                outputs.sequences, skip_special_tokens=True
            )
            # Placeholder: map string → class index outside or customize here.
            raise NotImplementedError("Multiclass decoding not implemented.")

        return pred

    @staticmethod
    def focal_loss(logits, labels, gamma=2.0, alpha_weights=None):
        probs = torch.nn.functional.softmax(logits, dim=-1)  # Convert logits to probabilities
        targets_one_hot = torch.nn.functional.one_hot(labels, num_classes=logits.size(-1)).float()  # One-hot encoding
        ce_loss = -targets_one_hot * torch.log(probs)  # Cross-entropy loss
        loss = (1 - probs) ** gamma * ce_loss  # Apply focal scaling
        if alpha_weights is not None:
            loss *= alpha_weights  # Weight per class
        return loss.mean()

    def forward_dst_readout(self, batch: HeteroData, entity_table: NodeType, dst_table: NodeType, ) -> Tensor:
        if self.id_awareness_emb is None:
            raise RuntimeError("id_awareness must be set True to use forward_dst_readout")
        seed_time = batch[entity_table].seed_time
        if self.use_rt_encoder:
            # Use encode method which handles RT
            x_dict, _ = self.encode(batch, entity_table)
        else:
            x_dict = self.encoder(batch.tf_dict)
            # Add ID-awareness to the root node
            x_dict[entity_table][: seed_time.size(0)] += self.id_awareness_emb.weight
            rel_time_dict = self.temporal_encoder(seed_time, batch.time_dict, batch.batch_dict)
            for node_type, rel_time in rel_time_dict.items():
                if node_type in x_dict:
                    x_dict[node_type] = x_dict[node_type] + rel_time
        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)
        if not self.use_rt_encoder:
            x_dict = self.gnn(x_dict, batch.edge_index_dict)
        return self.head(x_dict[dst_table])

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        return trainable_params, all_param
