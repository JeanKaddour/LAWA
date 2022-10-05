#!/bin/bash
avgmethods=(swa)
avg_saved_checkpoints_end=(30 40 50 60 70 80 90)

# unaveraged model checkpoints (i.e,. don't use LAWA)
for end in "${avg_saved_checkpoints_end[@]}"; do
    python imagenet.py --evaluate \
        --resume=/home/jean/projects/twa/save/imagenet/2022_09_20_23_25_55_185215/checkpoint$end.pt \
        -a resnet50 --dist-url tcp://127.0.0.1:2345 --dist-backend nccl --multiprocessing-distributed --world-size 1 \
        --rank 0 /home/jean/data/imagenet/ --avg_end_idx="$end"
done

# averaged model checkpoints (i.e., use LAWA)
ks=(4) # model at avg_end_idx is included (not excluded) in the averaging
for k in "${ks[@]}"; do
  for end in "${avg_saved_checkpoints_end[@]}"; do
      for avg in "${avgmethods[@]}"; do
        declare -i start=$end-$k
        # the resume checkpoint has to be valid for the script to work
        # however, it will be ignored/re-loaded with the one at avg_start_idx if averaging is enabled
        python imagenet.py --evaluate \
            --resume=/home/jean/projects/twa/save/imagenet/2022_09_20_23_25_55_185215/checkpoint10.pt \
            -a resnet50 --dist-url tcp://127.0.0.1:2345 --dist-backend nccl --multiprocessing-distributed --world-size 1 \
            --rank 0 /home/jean/data/imagenet/ --avg_dir=/home/jean/projects/twa/save/imagenet/2022_09_20_23_25_55_185215 \
            --avg_start_idx="$start" \
            --avg_end_idx="$end" \
            --avg_method="$avg" \
            --avg=True
      done
  done
done