#!/bin/bash
avgmethods=(swa)
avg_saved_checkpoints_end=(80 90 100 110 120 130 140 150 160 170 180 190)
cd fairseq

# unaveraged model checkpoints (i.e,. don't use LAWA)
# wandb will log them at step = avg-start-idx * num_steps_per_epoch

for start in "${avg_saved_checkpoints_start[@]}"; do
    for avg in "${avgmethods[@]}"; do
      fairseq-validate /home/jean/projects/fairseq/data-bin/wikitext-103 \
      --path /home/jean/projects/fairseq/multirun/2022-09-13/17-01-07/0/checkpoints/checkpoint$start.pt \
      --task masked_lm --batch-size 16 --wandb-project transformers --log-format tqdm --valid-subset train,valid,test \
      --skip-invalid-size-inputs-valid-test --avg-start-idx $start
    done
done

# averaged model checkpoints (i.e., use LAWA)
# you have to choose a valid checkpoint for ckpt_path
# however, it does not matter which one; the script will override the first model to be the one at $start
ckpt_path=/home/jean/projects/fairseq/multirun/2022-09-13/17-01-07/0/checkpoints/checkpoint70.pt
# averaged checkpoints
ks=(9) # model at avg-end-idx is included (not excluded) in the averaging
for k in "${ks[@]}"; do
  for end in "${avg_saved_checkpoints_end[@]}"; do
      for avg in "${avgmethods[@]}"; do
        declare -i start=$end-$k
        fairseq-validate /home/jean/projects/fairseq/data-bin/wikitext-103 \
        --path $ckpt_path \
        --task masked_lm --batch-size 16 --wandb-project transformers --log-format tqdm --valid-subset train,valid,test \
        --skip-invalid-size-inputs-valid-test --avg --avg-start-idx $start --avg-end-idx $end  --avg-method $avg
      done
    done
done

