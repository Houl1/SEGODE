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
       
3. **Data-scarce link prediction**: 

       python3 train.py --time_steps 14 --dataset wiki --gpu 0 --batch_size 1024 --tasktype data_scarce --scare_snapshot 9,10,11
       
# Data Sets
| **Data Set** | **Nodes** | **Edges** | **Min. Edges** | **Max. Edges** |**Snapshots**|
|:----:|:----:| :----: | :----: |:----: |:----: |
| Email | 1,891 | 5,870 | 102 | 2,861 | 6 |
| UCI  | 1,899 | 15,675 | 291 | 9,015 | 7 |
| Bitcoin | 3,783 | 14,380 | 945 | 3,113 | 8 |
| Reality | 6,809 | 16,623 | 607 | 1,351 | 18 |
| Wiki | 8,227 | 32,856 | 1,045 | 2,744 | 14|
| Reddit | 10,985 | 292,222 | 9,015 | 11,869 | 27 |
| Math | 24,740 | 230,956 | 4,899 | 13,162 | 26 | 

[Email](http://networkrepository.com/email-dnc.php): Email-dnc (Email) is a network of email exchanges between members of the U.S. Democratic National Committee in 2016.\
[UCI](http://konect.cc/networks/opsahl-ucsocial/): UCI records interactions between students at the University of California, Irvine, who post online in a forum.\
[Bitcoinalpha](http://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html): Bitcoinalpha (Bitcoin) is a rating network for the Bitcoin Alpha bitcoin trading platform.\
[Reality](http://networkrepository.com/ia-reality-call.php): Reality-call (Reality) is a user communication network from data from a user communication experiment at MIT.\
[Wikipedia](http://snap.stanford.edu/jodie/wikipedia.csv): Wikipedia (Wiki) is a network of Wikipedia page co-editors.\
[Reddit](http://snap.stanford.edu/jodie/reddit.csv): Reddit is a network of co-editors of posts on Reddit.\
[Math](http://snap.stanford.edu/data/sx-mathoverflow.html): Math is an interactive network of users on the Stack Exchange website Math Overflow.\
