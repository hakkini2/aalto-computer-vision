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


