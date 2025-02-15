### Official Code for "Consistency Training by Synthetic Question Generation for Conversational Question Answering" [ACL 2024](https://aclanthology.org/2024.acl-short.57.pdf) 

Note: I made sure that the code is not buggy right now and that it runs smoothly, but it may also be! Please note that I will add the code for training the Conversational Question Generation Module in the next few days.


#### Running

For Running CoTaH may simply run (change parameters by your will):

```
python Cotah_Bert.py --model_name CoTaH-BERT \
                       --S 2 \
                       --kl_ratio 2. \
                       --tau 6 \
                       --batch_size 6 \
                       --accumulation_steps 1 \
                       --seed 1000 \
                       --dist_type uniform \
                       --use_sim_threshold True \
                       --source_directory .   
```

Of course, They have a default value so may simply run:

```
python Cotah_Bert.py   
```

For Running BERT:

```
python Bert.py --model_name BERT \
               --batch_size 6 \
               --accumulation_steps 1 \
               --seed 1000 \
               --source_directory .   
```
