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
