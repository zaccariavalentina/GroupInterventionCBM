
## 💡 Usage

### N-CMAPSS: 
<u>Train</u> a **CEM** model with a CNN backbone on the NCMAPSS dataset. 
 
```bash
python ncmapss_experiments/ncmapss_train.py \
    --seed 42 \
    --data-path /mnt/disk1/valentina_zaccaria/ncmapss/ \
    --output-dir ./output/cemseparate/seed42/ \
    --model-type cnn_cem \
    --concepts '["Fan-E", "Fan-F", "LPC-E", "LPC-F", "HPC-E", "HPC-F", "LPT-E", "LPT-F", "HPT-E", "HPT-F"]' \
    --train_n_ds '["01-005", "04", "05", "07"]' \
    --test_n_ds '["01-005", "04", "05", "07"]' \
    --train_units '[1, 2, 3, 4, 5, 6]' \
    --test_units '[7, 8, 9, 10]'
```
**Note**: change `--model-type` for a different architecture. 


<u>Evaluate</u> a trained CEM model with grouped interventions. 
```bash
python ncmapss_experiments/ncmapss_group_int_eval.py \
    --checkpoint ./ncmapss_experiments/your/checkpoint/path/epoch\=24-val_loss\=0.06.ckpt \
    --data_path /your/data/path \
    --policy_type optimized \
    --budget 3.5 \
```

### CheXpert
<u>Train</u> a **Joint CBM with sigmoid** on the CheXpert dataset: 

```bash
python chexpert_experiments/train.py \
    --seed 42 \
    --arch XtoCtoY_sigmoid 
```

```bash
python chexpert_experiments/chexpert_group_int_eval.py \
    --bottleneck_type independent
```
