### Official Code for "Consistency Training by Synthetic Question Generation for Conversational Question Answering" Paper 

Note: I made sure that code is not buggy right now, and it runs smoothly, but it may also be! Please note that I will be adding the code for training the Conversational Question Generation Module in the next days.


#### Running

for Running CoTaH may simple run (change parameters by your will):

```
python Cotah_Bert.py --model_name CoTaH-BERT \
                       --S 2 \
                       --kl_ratio 2. \
                       --gamma .8 \
                       --M 10 \
                       --tau 6 \
                       --batch_size 6 \
                       --accumulation_steps 1 \
                       --seed 1000 \
                       --dist_type uniform \
                       --use_sim_threshold True \
                       --source_directory .   
```

for Running BERT:

```
python Bert.py --model_name BERT \
               --batch_size 6 \
               --accumulation_steps 1 \
               --seed 1000 \
               --source_directory .   
```
