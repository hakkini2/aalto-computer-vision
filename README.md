# SAM on MSD data



### How to run

First, the 2D slices of the 3D images must be created with:

```
python3 create-2d-slices.py --data_txt_path <path to txt file>
```

where \<path to txt file\> is the path to the txt-file describing the data splits to train, val and test. Notice the format of the file name: for example, we have files *MSD_Task03_Liver_train.txt*, *MSD_Task03_Liver_val.txt* and *MSD_Task03_Liver_test.txt* in the folder *./dataset/*. Then  \<path to txt file\> is *'./dataset/MSD_Task03_Liver'*.

Then, SAM-inference.py can be run for all MSD tasks with:

```
python3 SAM-inference.py
```


The program can also be run for individual organs with:

```
python3 SAM-inference.py --organ <organ>
```

where \<organ\> is one of the MSD tasks: *liver, lung, pancreas, hepaticvessel, spleen, colon*.


For saving the results, the program create-empty-3d-masks.py must be run individually for all the MSD tasks before running SAM-inference.py. When running, change the value in argument --data_txt_path to change the current MSD task, for example:

```
python3 create-empty-3d-masks.py --data_txt_path './dataset/MSD_Task07_Pancreas'
```

Then, SAM-inference.py can be run with the argument --save_results:

```
python3 SAM-inference.py --save_results True
```