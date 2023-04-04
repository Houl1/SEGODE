# SEGODE
This repository includes the source code and data sets used in our paper:
Structure-Enhanced Graph Neural ODE Network for Temporal Link Prediction

# SEGOD Requirements
    - Python>= 3.8
    - torch==1.8.0
    - torch-cluster==1.6.0
    - torch-geometric==2.2.0
    - torch-scatter==2.0.9
    - torch-sparse==0.6.13
    - torch-spline-conv==1.2.1
    - torchrec==0.3.2
    - torchdiffeq==0.2.3
   
# Directory
    
    SEGODE/    
        data/                        (data sets)  
            bitcoin/  
                gdv/                       (Graphlet Degree Vector)  
                no_repetition_selfloop/    (snapshot without repetition and selfloop)    
            email/
            ......
        eval/                        (link prediction evaluation tasks)  
        model_checkpoints/           (Storage Model) 
        models/                      (Model specific implementation)
            model.py                 (SEGODE model) 
            layers.py                (Layers needed for SEGODE model building)
        orca/                        (Code for calculating Graphlet Degree Vector via C++ implementation)
        utils/                       (utility files)
        train.py                     (main file used to train single-step and scarce-data link prediction)  
        train_multisteps.py          (main file used to train multi-step link prediction)

# Commands
This project has several tasks, including: **Single-step link prediction**, **Multi-step link prediction** and **Scarce-data link prediction**. Thus, the corresponding Python commands are:
1. **Single-step link prediction**: 

       python3 train.py --time_steps 8 --dataset bitcoinotc --gpu 0 --batch_size 1024 --tasktype siglestep

2. **Multi-step link prediction**: 

       python3 train_multisteps.py --time_steps 14 --dataset wiki --gpu 0 --batch_size 1024 --tasktype multisteps
       
3. **Scarce-data link prediction**: 

       python3 train.py --time_steps 14 --dataset wiki --gpu 0 --batch_size 1024 --tasktype data_scarce --scare_snapshot 9,10,11
       
 
    
