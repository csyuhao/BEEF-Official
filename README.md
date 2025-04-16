# A Wolf in Sheep's Clothing: Unveiling a Stealthy Backdoor Attack in Subgraph Federated Learning

## Abstract

Subgraph Federated Learning (FL) is an emerging paradigm for node classification tasks, where subgraphs of a global graph are distributed across multiple devices to mitigate data leakage risks.
Similar to other FL systems, subgraph FL faces significant security challenges, particularly from backdoor attacks, an area that remains underexplored.
In these attacks, adversaries often adapt image domain techniques and use a two-phase mechanism to generate backdoored models.
However, in subgraph FL, such attacks typically result in substantial parameter discrepancies between backdoored and normal models, undermining the attack's stealthiness.
To tackle this challenge, we propose BEEF, a Backdoor attack with an End-to-End Framework designed for effectiveness, stealth, and durability.
Unlike conventional methods, BEEF incorporates a dedicated trigger generator, which is jointly trained with a backdoored model.
To increase its stealthiness, BEEF crafts adversarial perturbations as triggers that provoke misclassification while leaving the model’s parameters entirely untouched.
Furthermore, by calibrating a subset of low-salience parameters associated with backdoor activation, BEEF ensures stable performance and sustained effectiveness across FL rounds.
Comprehensive evaluations across eight datasets, nine models (including four robust models tailored for adversarial examples), five state-of-the-art attacks (e.g., AutoAdapt, NDSS 2024), and eleven aggregation methods (e.g., FLAME, USENIX Security 2022; MESAS, CCS 2023; FreqFed, NDSS 2023) demonstrate BEEF's effectiveness in deceiving GNNs while maintaining minimal impact on normal data performance.
For instance, in an FL system with 20\% compromised clients, BEEF achieves a 98.74\% attack success rate (ASR) and 82.63\% accuracy on the PubMed dataset using the FLAME aggregation method, significantly outperforming the best baseline, which achieves only 27.37\% ASR.  
Additionally, we adapt BEEF to federated graph classification tasks, broadening its applicability and practicality.


## Requirements

The code can run in GPU environments.
- Windows 11 (or Ubuntu may)
- Python = 3.10.11
- PyTorch = 2.0.0
- PyTorch-Geometric = 2.5.3
- Other Python libraries listed in ```secgnn.yml```

Please install packages using the following command:
```bash
conda env create -f secgnn.yml --name secgnn
```

## File Architecture

```
+---GraphClassification         Backdoor attacks on graph classification tasks
|   |     |─ data.py
|   |     |─ main.py
|   |     |─ model.py
|   |     └─ utils.py
|   |
|   |---beef
|         └─ generator.py
|---NodeClassification          Backdoor attacks on node classification tasks
    |     |─ main.py
    |     └─ utils.py
    +---attacks
    |     └─ beef.py
    +---clients
    |     |─ client.py
    |     └─ utils.py
    +---dataset
    |     |─ amazon.py
    |     |─ coauther.py
    |     |─ ogba.py
    |     |─ partition.py
    |     └─ utils.py
    +---defenses
    |     └─ fedavg.py
    +---models
    |     |─ gcn.py
    |     └─ utils.py
    +---servers
    |     └─ server.py
    +---triggers
    |     |─ generator.py
    |     |─ gen_trigger.py
    |     |─ gta.py
    |     |─ heuristic.py
    |     |─ position.py
    |     |─ trojan_net.py
    |     |─ ugba.py
    |     |─ utils.py
```

### Node Classification

```bash
cd NodeClassification
```

In node classification tasks, we utilize ```WandB``` to monitor the experimental workflow.
However, network instability may disrupt this process.
To ensure uninterrupted execution, you can run WadnB in offline mode using the following command:

```bash
wandb offline
```

This command temporarily disables synchronization with the WadnB server, allowing the experiment to proceed locally without relying on an internet connection.

1. BEEF backdoor attack on the Cora dataset

```bash
python main.py --seed=1314 --num_clients=10 --num_chosen_clients=10 --num_malicious=2 --lr=0.01 --weight_decay=0.0 --num_rounds=100 --attack_name='beef' --num_hidden_generator=128 --magnitude_thresh_ratio=0.05 --target_class=0 --num_start_round=0 --trigger_type='beef' --trigger_position='learn_cluster_degree' --outer_epochs=100 --dataset='cora' --data_distribution='uniform' --model='GCN' --local_epochs=1 --n_round=10 --tau=0.01 --defense_name='fedavg' --group_name='Cora-GCN-FedAvg-BEEF(1314)'
```

2. BEEF backdoor attack on the Reddit dataset

```bash
python main.py --seed=1314 --num_clients=10 --num_chosen_clients=10 --num_malicious=2 --lr=0.01 --weight_decay=0.0 --num_rounds=100 --attack_name='beef' --num_hidden_generator=256 --magnitude_thresh_ratio=0.05 --target_class=0 --num_start_round=0 --trigger_type='beef' --trigger_position='learn_cluster_degree' --outer_epochs=100 --dataset='reddit' --data_distribution='uniform' --model='GCN' --local_epochs=1 --num_hidden=128 --n_round=10 --tau=0.01 --defense_name='fedavg' --group_name='Reddit-GCN-FedAvg-BEEF(1314)'
```

### Graph Classification

```bash
cd GraphClassification
```

1. BEEF backdoor attack on the MUTAG dataset

```bash
python main.py --dataset=MUTAG --attack_method=BEEF --num_agents=10 --num_corrupt=2 --epochs=100 --trigger_size=5
```