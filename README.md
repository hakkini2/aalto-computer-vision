# SAM on MSD data



### How to run

First, the 2D slices of the 3D images must be created with:

```
python3 create-2d-slices.py
```

Then, SAM-inference.py can be run for all MSD tasks with:

```
python3 SAM-inference.py
```


The program can also be run for individual organs with:

```
python3 SAM-inference.py --organ <organ>
```

where \<organ\> is one of the MSD tasks: *liver, lung, pancreas, hepaticvessel, spleen, colon*.


For saving the results, program create-empty-3d-masks.py must be run for all the tasks before running SAM-inference.py. When running, change the value in argument --data_txt_path when running, for example:

```
python3 create-empty-3d-masks.py --data_txt_path './dataset/MSD_Task07_Pancreas'
```

Then, SAM-inference.py can be run with the argument --save_results:

```
python3 SAM-inference.py --save_results True
```