HY4_HOT seed scores, a file a model folder (serve/server.py hy4_hot_seed), from upstream Maya v1.1.75.
Maya-XS and Maya-M: learned on their own packs (upstream's Jupiter, V100, 16 greedy prompts).
Maya-S and Maya-L: copies of the Maya-XS file (XS/M scores correlate 0.98) until learned on their own packs - replace them then.
A file of the PC's own in data/ wins (data/hy4-expert-hot-Hy4-Flash-Maya-S.bin: ours, learned on 4x Tesla T4).
