python train_transformer_hxmt_32s.py --train-csv ../data/hxmt_merged_dataset_HE_v3.0.csv --epochs=500 --batch-size=256 --learning-rate=1e-4 --patience=100 --device='cuda' --channel-min 8 --channel-max 162
python plot_transformer_results.py   --window=32  --block-duration=32    --month-stride-days 30     --max-evals 12     --metrics-csv artifacts/monthly_eval.csv --min-ch=8 --max-ch=162
