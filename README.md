# PyTorch Distributed Training 🌐

Production recipes for distributed PyTorch training.

## Parallelism Strategies

| Strategy | GPU Memory | Speed | Best For |
|----------|-----------|-------|----------|
| DDP | 1× | Linear | <30B params |
| FSDP | 1/N | 0.8× | 30-100B params |
| DeepSpeed ZeRO-3 | 1/N | 0.7× | 100B+ params |
| Pipeline Parallel | 1/P | 0.9× | Very large models |

## Quick Start

```python
from pytorch_distributed import DistributedTrainer

trainer = DistributedTrainer(
    model=model,
    strategy="fsdp",
    num_gpus=8,
)
trainer.fit(train_dataset, epochs=10)
```

## License

MIT