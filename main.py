import argparse
import json
import math
import copy
import os
import time
import wandb
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from model import Model
from text_embedder import TextEmbedding
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from tqdm import tqdm

from relbench.base import Dataset, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task
from utils import task_info
from rt_adapter import create_rt_loader
import warnings
warnings.filterwarnings(
    "ignore",
    message="cuDNN SDPA backward got grad_output.strides() != output.strides()",
)
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['CURL_CA_BUNDLE'] = ''  # huggingface connection issue


def get_batch_labels(batch, entity_table, device, task=None):
    """
    Unified label extractor for:
      - HeteroData batches (original Rel-LLM)
      - RT batches (dict from RelationalDataset)
    """
    # ---------- RT batch (dict from RT sampler) ----------
    if isinstance(batch, dict):
        # Preferred path: use RT's explicit class_value_idxs + is_targets
        if "class_value_idxs" in batch and "is_targets" in batch:
            is_targets = batch["is_targets"].bool()          # (B, S)
            class_value_idxs = batch["class_value_idxs"]     # (B, S)

            # Flatten targets across batch
            y_all = class_value_idxs[is_targets]             # (N_targets,)

            if y_all.numel() == 0:
                raise RuntimeError("No targets found in RT batch (is_targets has no True).")

            # true_batch_size = number of seed examples in this batch
            true_bs = int(batch.get("true_batch_size", y_all.shape[0]))
            y_all = y_all[:true_bs]                          # (true_bs,)

            # Map to float labels depending on task type
            if task is not None and task.task_type == TaskType.BINARY_CLASSIFICATION:
                # For binary tasks, class_value_idxs should already be 0/1.
                # Just clamp/sanitize to be safe.
                y = torch.clamp(y_all, 0, 1).float()
            else:
                # Other task types: keep as-is (you can customize later)
                y = y_all.float()

            return y.to(device)

        # Fallback: try a direct tensor field if someone stored labels there
        for key in ["targets", "y", "labels"]:
            if key in batch:
                return batch[key].to(device).float()

        raise KeyError(
            f"Could not find labels in RT batch. "
            f"Tried 'class_value_idxs' + 'is_targets' or ['targets', 'y', 'labels']. "
            f"Available keys: {list(batch.keys())}"
        )

    # ---------- Original HeteroData path ----------
    # batch[entity_table].y is already 0/1 or numeric
    return batch[entity_table].y.to(device).float()


@torch.no_grad()
def test(loader, demo_info=None, split: str = "test") -> np.ndarray:
    model.eval()
    pred_list = []
    for test_batch in tqdm(loader):
        # Handle RT batch format (dict) vs HeteroData
        if model.use_rt_sampler:
            # RT batch is already a dict, move to device
            for k, v in test_batch.items():
                if isinstance(v, torch.Tensor):
                    test_batch[k] = v.to(device)
        else:
            test_batch = test_batch.to(device)
        pred = model(test_batch, task.entity_table, demo_info=demo_info, inference=True, split=split)
        if task.task_type == TaskType.BINARY_CLASSIFICATION and len(pred_list) == 0:
            # Trace label distribution on first batch only (if labels are available)
            try:
                y = get_batch_labels(test_batch, entity_table=task.entity_table, device=pred.device, task=task)
                pos = (y > 0.5).sum().item()
                neg = (y <= 0.5).sum().item()
                print("[TRACE] y stats:", y.min().item(), y.max().item(), y.float().mean().item(), y.numel(), flush=True)
                print(f"[TRACE] y counts: pos={pos} neg={neg}", flush=True)
            except Exception as e:
                print(f"[TRACE] y stats unavailable: {e}", flush=True)
        if task.task_type == TaskType.REGRESSION:
            assert clamp_min is not None and clamp_max is not None
            pred = torch.clamp(pred, clamp_min, clamp_max)

        if (args.model_type == 'gnn' or args.output_mlp) and task.task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTILABEL_CLASSIFICATION]:
            pred = torch.sigmoid(pred)  # normalize to between 0 and 1

        pred = pred.view(-1) if len(pred.size()) > 1 and pred.size(1) == 1 else pred
        if len(pred_list) == 0:
            print("[TRACE] pred stats:", pred.min().item(), pred.max().item(), pred.mean().item(), pred.std().item(), flush=True)
        pred_list.append(pred.detach().cpu())
    return torch.cat(pred_list, dim=0).numpy()


if __name__ == '__main__':
    # only classification tasks # todo: different tasks in the same dataset are different training sizes?
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="rel-stack")
    parser.add_argument("--task", type=str, default="user-engagement")
    parser.add_argument("--cache_dir", type=str, default=os.path.expanduser("~/.cache/relbench_examples"), )
    parser.add_argument("--debug", action='store_true')

    # GNNs
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--aggr", type=str, default="sum")
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_neighbors", type=int, default=128)
    parser.add_argument("--temporal_strategy", type=str, default="uniform", choices=['uniform', 'last'])
    parser.add_argument("--text_embedder", type=str, default='glove', choices=['glove', 'mpnet'])
    parser.add_argument("--text_embedder_path", type=str, default="./cache")

    # RT encoder
    parser.add_argument("--use_rt_encoder", action='store_true', help="Use Relational Transformer encoder instead of GNN")
    parser.add_argument("--rt_num_blocks", type=int, default=4, help="Number of RT transformer blocks")
    parser.add_argument("--rt_d_model", type=int, default=256, help="RT model dimension (pretrained uses 256)")
    parser.add_argument("--rt_d_text", type=int, default=384, help="RT text embedding dimension")
    parser.add_argument("--rt_num_heads", type=int, default=8, help="Number of attention heads in RT")
    parser.add_argument("--rt_d_ff", type=int, default=1024, help="RT feed-forward dimension (pretrained uses 1024)")
    parser.add_argument("--rt_pretrained_path", type=str, default=None, help="Path to RT pretrained checkpoint")

    # Debug / tracing
    parser.add_argument("--trace_shapes", action="store_true", help="Print tensor shapes/values for LLM prefix tracing")

    # LLMs
    # huggingface-cli download --resume-download gpt2 --local-dir gpt2
    # huggingface-cli download --resume-download deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --local-dir /ai/design/RelGraph/DeepSeek-R1-Distill-Qwen-32B
    parser.add_argument("--model_type", type=str, default="meta-llama/Llama-3.2-1B",
                        choices=['deepseek-ai/DeepSeek-R1-Distill-Qwen-32B', "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.2-1B",
                                 "meta-llama/Llama-3.2-3B-Instruct"])
    parser.add_argument("--llm_frozen", action='store_true')
    parser.add_argument("--output_mlp", action='store_true')
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_demo", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=1)
    parser.add_argument('--loss_class_weight', nargs='+', type=float, default=None)

    # training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--pretrain", action='store_true')
    parser.add_argument("--pretrain_epochs", type=int, default=200)
    parser.add_argument("--val_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)  # RT's default is 32; GNN can use larger (e.g., 256)
    parser.add_argument("--val_size", type=int, default=None)  # default 512 for GNN
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0001)  # default 0.005 for GNN
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_num_threads(1)

    #############################################
    # get data and task information
    #############################################
    dataset: Dataset = get_dataset(args.dataset, download=True)  # get dataset (database + temporal splitting times)
    db = dataset.get_db()  # get database
    print('Table names: ', list(db.table_dict.keys()))
    print('Begin time: ', db.min_timestamp, 'End time: ', db.max_timestamp)
    print('Val time: ', dataset.val_timestamp, 'Test time: ', dataset.test_timestamp)
    task = get_task(args.dataset, args.task, download=True)
    task.name = args.task

    # notebook: https://colab.research.google.com/github/snap-stanford/relbench/blob/main/tutorials/train_model.ipynb
    stypes_cache_path = Path(f"{args.cache_dir}/{args.dataset}/stypes.json")
    try:  # configurate stype (modality) of each column, e.g., numerical/timestamp/categorical/text_embedded
        with open(stypes_cache_path, "r") as f:
            col_to_stype_dict = json.load(f)
        for table, col_to_stype in col_to_stype_dict.items():
            for col, stype_str in col_to_stype.items():
                col_to_stype[col] = stype(stype_str)
    except FileNotFoundError:
        col_to_stype_dict = get_stype_proposal(db)
        Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(stypes_cache_path, "w") as f:
            json.dump(col_to_stype_dict, f, indent=2, default=str)

    # build heterogeneous and temporal graphs `data`
    # sentence_transformer: https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
    os.makedirs(args.text_embedder_path, exist_ok=True)
    text_embedder = TextEmbedding(args.text_embedder, args.text_embedder_path, device=device)
    data, col_stats_dict = make_pkey_fkey_graph(db, col_to_stype_dict=col_to_stype_dict, text_embedder_cfg=TextEmbedderConfig(text_embedder=text_embedder, batch_size=256),
                                                cache_dir=f"{args.cache_dir}/{args.dataset}/materialized")

    # 'num_neighbors' -> the number of neighbors sampled per node (e.g., [64, 32, 16]), 'num_sampled_nodes' -> the total number of nodes sampled per layer (hop)
    out_channels, loss_fn, tune_metric, higher_is_better, clamp_min, clamp_max = task_info(task)
    
    # Determine entity table
    table = task.get_table("train")
    table_input = get_node_train_table_input(table=table, task=task)
    entity_table = table_input.nodes[0]
    print('Entity table: ', entity_table)
    
    # RT encoder REQUIRES RT's sampler - NEVER use NeighborLoader when RT encoder is enabled
    loader_dict: Dict = {}
    rt_bridge_dict: Dict = {}
    
    if args.use_rt_encoder:
        # REQUIRED: Use RT's sampler with BFS (NeighborLoader is NOT used)
        print("Using RT's sampler with BFS traversal (RT encoder REQUIRES RT sampler)")
        for split in ["train", "val", "test"]:
            rt_batch_size = args.batch_size if (split == 'train' or args.val_size is None) else (args.val_size or args.batch_size)
            loader, bridge = create_rt_loader(
                task=task,
                dataset_name=args.dataset,
                entity_table=entity_table,
                split=split,
                batch_size=rt_batch_size,
                seq_len=1024,  # RT's default seq_len
                max_bfs_width=256,  # RT's default max_bfs_width
                embedding_model="all-MiniLM-L12-v2",  # RT's default
                d_text=args.rt_d_text,
                num_workers=args.num_workers,
                seed=args.seed,
            )
            loader_dict[split] = loader
            rt_bridge_dict[split] = bridge
        args.val_steps = min(args.val_steps, len(loader_dict['train']))
    else:
        # Use NeighborLoader (original Rel-LLM approach - GNN only, no RT encoder)
        print("Using NeighborLoader (GNN path, RT encoder disabled)")
        for split in ["train", "val", "test"]:
            table = task.get_table(split)
            table_input = get_node_train_table_input(table=table, task=task)
            bs = args.batch_size if (split == 'train' or args.val_size is None) else args.val_size
            loader_dict[split] = NeighborLoader(data, num_neighbors=[int(args.num_neighbors / 2 ** i) for i in range(args.num_layers)], time_attr="time", input_nodes=table_input.nodes,
                                                input_time=table_input.time, transform=table_input.transform, batch_size=bs, temporal_strategy=args.temporal_strategy,
                                                shuffle=split == "train", num_workers=args.num_workers, persistent_workers=args.num_workers > 0, pin_memory=True)  # TODO: bidirectional
        args.val_steps = min(args.val_steps, len(loader_dict['train']))

    #############################################
    # model training
    #############################################
    # RT config
    rt_config = None
    if args.use_rt_encoder:
        rt_config = {
            'num_blocks': args.rt_num_blocks,
            'd_model': args.rt_d_model,
            'd_text': args.rt_d_text,
            'num_heads': args.rt_num_heads,
            'd_ff': args.rt_d_ff,
            'pretrained_path': args.rt_pretrained_path,
        }
    
    model = Model(
        data,
        col_stats_dict,
        args.num_layers,
        channels=args.channels,
        out_channels=out_channels,
        aggr=args.aggr,
        dropout=args.dropout,
        model_type=args.model_type,
        llm_frozen=args.llm_frozen,
        output_mlp=args.output_mlp,
        max_new_tokens=args.max_new_tokens,
        alpha=args.loss_class_weight,
        num_demo=args.num_demo,
        dataset=args.dataset,
        task=task,
        use_rt_encoder=args.use_rt_encoder,
        rt_config=rt_config,
        text_embedder=text_embedder,
        trace_shapes=args.trace_shapes,
    ).to(device)
    
    # Pass RT bridges to model if using RT sampler
    if args.use_rt_encoder and 'rt_bridge_dict' in locals():
        model.rt_bridge_dict = rt_bridge_dict
        model.use_rt_sampler = True
    else:
        model.use_rt_sampler = False
    
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    if args.wd != 0:  # weight decay should not be applied to bias terms and LayerNorm parameters
        optimizer = torch.optim.AdamW([{'params': [p for n, p in model.named_parameters() if "bias" not in n and "LayerNorm" not in n], 'weight_decay': args.wd},
                                       {'params': [p for n, p in model.named_parameters() if "bias" in n or "LayerNorm" in n], 'weight_decay': 0.0}], lr=args.lr, betas=(0.9, 0.95))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max' if higher_is_better else 'min', factor=0.8, patience=100)
    trainable_params, all_param = model.print_trainable_params()
    print(f"trainable params: {trainable_params / 1e6:.2f}M || all params: {all_param / 1e6:.2f}M || trainable: {100 * trainable_params / all_param:.4f}%")

    # #############################################
    # # pretraining
    # #############################################
    state_dict = None
    if args.pretrain:
        if not args.debug:  # rename the project if init failure
            run = wandb.init(project='rel-LLM-zero', name=f'{args.dataset}_{args.task}', id=f"pretrain_run_{args.dataset}_{args.task}", resume="allow")
        pretrain_steps = 0
        best_val_metric = -math.inf if higher_is_better else math.inf
        pretrain_batch_start_time = time.time()  # Initialize batch timing for pretrain
        for epoch in range(1, args.pretrain_epochs + 1):
            loss_accum = count_accum = 0
            tq = tqdm(loader_dict["train"], total=len(loader_dict["train"]))
            for batch_idx, batch in enumerate(tq):
                try:
                    model.train()
                    # Handle RT batch format (dict) vs HeteroData
                    if model.use_rt_sampler:
                        # RT batch is already a dict, move to device
                        for k, v in batch.items():
                            if isinstance(v, torch.Tensor):
                                batch[k] = v.to(device)
                        nums_samples = batch.get('true_batch_size', batch['node_idxs'].shape[0])
                        split = "train"
                    else:
                        batch = batch.to(device)
                        nums_samples = batch[entity_table].y.size(0)
                        split = None
                    optimizer.zero_grad()
                    loss = model.pretrain(batch, task.entity_table, split=split)
                    loss.backward()
                    optimizer.step()
                except torch.OutOfMemoryError:
                    print("Skipping batch due to CUDA out of memory error")
                    torch.cuda.empty_cache()  # Free up cached memory
                    continue
                pretrain_steps += 1
                loss_accum += loss.detach().item() * nums_samples
                count_accum += nums_samples
                train_loss = loss_accum / count_accum
                
                # Time logging
                batch_end_time = time.time()
                if pretrain_steps == 1:
                    pretrain_batch_start_time = batch_end_time
                batch_time = batch_end_time - pretrain_batch_start_time if pretrain_steps > 1 else 0.0
                steps_per_sec = 1.0 / batch_time if batch_time > 0 else 0.0
                pretrain_batch_start_time = batch_end_time
                
                summary = {
                    'loss': train_loss, 
                    'lr': optimizer.param_groups[-1]['lr'],
                    'batch_time': batch_time,
                    'steps_per_sec': steps_per_sec
                }
                if not args.debug:
                    for k, v in summary.items():
                        run.log({f'Pretrain/{k}': v}, step=pretrain_steps)  # Steps must be monotonically increasing
                tq.set_description(f'[Pretrain] Epoch/Step: {epoch:02d}/{pretrain_steps} | Train loss: {train_loss:.4f} | {steps_per_sec:.2f} steps/s')

                # zero-shot / few-shot evaluation
                if pretrain_steps % args.val_steps == 0:
                    # Get demo info
                    train_batch = next(iter(loader_dict["train"]))
                    if model.use_rt_sampler:
                        for k, v in train_batch.items():
                            if isinstance(v, torch.Tensor):
                                train_batch[k] = v.to(device)
                    else:
                        train_batch = train_batch.to(device)
                    demo = model.get_demo_info(train_batch, task.entity_table, split="train") if args.num_demo > 0 else None
                    
                    val_pred = test(loader_dict["val"], demo, split="val")
                    val_metrics = task.evaluate(val_pred, task.get_table("val"))
                    if not args.debug:
                        for k, v in val_metrics.items():
                            run.log({f'val/{k}': v}, step=pretrain_steps)

                    if (higher_is_better and val_metrics[tune_metric] >= best_val_metric) or (not higher_is_better and val_metrics[tune_metric] <= best_val_metric):
                        best_val_metric = val_metrics[tune_metric]
                        test_pred = test(loader_dict["test"], demo, split="test")
                        test_metrics = task.evaluate(test_pred, task.get_table("test", mask_input_cols=False))
                        if not args.debug:
                            for k, v in test_metrics.items():
                                run.log({f'test/{k}': v}, step=pretrain_steps)
                        state_dict = copy.deepcopy(model.state_dict())
                    scheduler.step(val_metrics[tune_metric])
                    print(f'[Eval] Epoch/Step: {epoch:02d}/{pretrain_steps} | Val: {val_metrics} | Best val/test: {best_val_metric:.4f}/{test_metrics[tune_metric]:.4f}')

    #############################################
    # Fine-tuning
    #############################################
    steps = 0
    if not args.debug:
        if args.pretrain:
            run.finish()
        # Create a new run with timestamp to avoid resuming old runs
        timestamp = int(time.time())
        run = wandb.init(entity="9race-stanford", project='rel-LLM', name=f'{args.dataset}_{args.task}', id=f"finetune_run_{args.dataset}_{args.task}_{timestamp}", resume="never")
    if state_dict is not None: model.load_state_dict(state_dict)  # load pretrained weights
    best_val_metric = -math.inf if higher_is_better else math.inf
    batch_start_time = time.time()  # Initialize batch timing for fine-tuning
    print(f"[DEBUG] Starting training loop, total batches: {len(loader_dict['train'])}")
    for epoch in range(1, args.epochs + 1):
        loss_accum = count_accum = 0
        print(f"[DEBUG] Epoch {epoch}, creating DataLoader iterator...")
        tq = tqdm(loader_dict["train"], total=len(loader_dict["train"]))
        print(f"[DEBUG] Starting to iterate over batches...")
        for batch_idx, batch in enumerate(tq):
            if batch_idx == 0:
                print(f"[DEBUG] Got first batch from DataLoader!")
            model.train()
            # Handle RT batch format (dict) vs HeteroData
            if model.use_rt_sampler:
                # IMPORTANT: keep RT batch on CPU; Model.encode will handle device moves
                if steps == 0:
                    print(f"[DEBUG] First batch received (RT), staying on CPU before encode...")
                nums_samples = batch.get('true_batch_size', batch['node_idxs'].shape[0])
                split = "train"
                if steps == 0:
                    print(f"[DEBUG] RT batch info: batch_size={nums_samples}, node_idxs shape={batch['node_idxs'].shape}")
            else:
                batch = batch.to(device)
                nums_samples = batch[entity_table].y.size(0)
                split = None
            optimizer.zero_grad()  # continued learning rate
            if steps == 0:
                print(f"[DEBUG] Starting forward pass...")
            if args.model_type == 'gnn' or args.output_mlp:
                output_pred = model(batch, task.entity_table, split=split)
                output_pred = (
                    output_pred.view(-1)
                    if len(output_pred.size()) > 1 and output_pred.size(1) == 1
                    else output_pred
                )

                # 🔑 Get labels correctly for RT vs GNN
                y = get_batch_labels(
                    batch,
                    entity_table=entity_table,
                    device=output_pred.device,
                    task=task,
                )

                loss = loss_fn(output_pred.float(), y)
            else:
                loss = model(batch, task.entity_table, split=split)
            if steps == 0:
                print(f"[DEBUG] Forward pass complete, loss={loss.item():.4f}")
            loss.backward()
            optimizer.step()

            steps += 1
            loss_accum += loss.detach().item() * nums_samples
            count_accum += nums_samples
            train_loss = loss_accum / count_accum
            
            # Time logging
            batch_end_time = time.time()
            if steps == 1:
                batch_start_time = batch_end_time
            batch_time = batch_end_time - batch_start_time if steps > 1 else 0.0
            steps_per_sec = 1.0 / batch_time if batch_time > 0 else 0.0
            batch_start_time = batch_end_time
            
            summary = {
                'loss': train_loss, 
                'lr': optimizer.param_groups[-1]['lr'],
                'batch_time': batch_time,
                'steps_per_sec': steps_per_sec
            }
            if not args.debug:
                for k, v in summary.items():
                    run.log({f'train/{k}': v}, step=steps)
            tq.set_description(f'[Train] Epoch/Step: {epoch:02d}/{steps} | Train loss: {train_loss:.4f} | {steps_per_sec:.2f} steps/s')
            if steps % args.val_steps == 0:
                val_pred = test(loader_dict["val"], split="val")
                val_metrics = task.evaluate(val_pred, task.get_table("val"))
                test_pred = test(loader_dict["test"], split="test")
                test_metrics = task.evaluate(test_pred, task.get_table("test", mask_input_cols=False))
                if not args.debug:
                    for k, v in val_metrics.items():
                        run.log({f'val/{k}': v}, step=steps)
                    for k, v in test_metrics.items():
                        run.log({f'test/{k}': v}, step=steps)
                if (higher_is_better and val_metrics[tune_metric] >= best_val_metric) or (not higher_is_better and val_metrics[tune_metric] <= best_val_metric):
                    best_val_metric = val_metrics[tune_metric]
                    state_dict = copy.deepcopy(model.state_dict())
                scheduler.step(val_metrics[tune_metric])
                print(f'[Eval] Epoch/Step: {epoch:02d}/{steps} | Val: {val_metrics} | Test: {test_metrics} | Best val: {best_val_metric:.4f}')

    #############################################
    # evaluation
    #############################################
    if state_dict is not None:
        model.load_state_dict(state_dict)
    val_pred = test(loader_dict["val"], split="val")
    val_metrics = task.evaluate(val_pred, task.get_table("val"))
    test_pred = test(loader_dict["test"], split="test")
    test_metrics = task.evaluate(test_pred, task.get_table("test", mask_input_cols=False))
    print(f"Best Val metrics: {val_metrics}")  # nuance due to sampling
    print(f"Best test metrics: {test_metrics}")
    for k, v in test_metrics.items():
        run.log({f'test/{k}': v}, step=steps + 1)
