for task in {matching,ranking};
  do for sp in {1,2,3,4};
    do for lr in {.00001,.00005,.000005};
      do CUDA_VISIBLE_DEVICES=1 python train_clip.py $sp $task --warmup 200 --clip_model ViT-L/14@336px --pad 1 --lr $lr --use_accelerate 0 --batch_size 1 --n_epochs 12;
    done;
  done;
done;