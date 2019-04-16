# Handwriting Detection

## Overview

binclass can use slices in order to detect a feature that only appears in a small part of the image, provided that the false positive rate is low enough that the page-wide false positive rate is acceptable. The included trained model targets monochrome document images with actual handwriting but was cross-trained on color photographs of documents, color scans, and even artwork depicting handwriting, which improved the accuracy on the target.

## Use

Put files in a directory (`testdir`, for example), and invoke the check on each file as follows.

```
for file in testdir/*.png ; do python3 ./binclass.py --predict-file "$file" --model-input examples/handwriting/handwriting_resnet.bin --slice-width 150 --slice-height 150 --slice-count-threshold-proportional 0.01 2> /dev/null > "$file".result.txt ; done ;
```

## Retraining

Classify page images into positive (with handwriting) and negative (without handwriting). Make slices of 150 pixels by 150 pixels of all pages.

```
for imgfile in *.png ; do convert "$imgfile" "$imgfile".pnm; pamdice -outstem pamdice_0_ -width 150 -height 150 "$imgfile".pnm; rm "$imgfile".pnm; SliceC=0; for slicefile in pamdice_0_*.p?m; do SliceC=`expr $SliceC + 1`; pnmtopng "$slicefile" > "$imgfile".Slice_`printf "%04d" $SliceC`.png ; rm "$slicefile"; done; done;
```

Manually classify slices from positive pages as positive or negative. Put all positive slices in one directory, put all negative slices in the other directory, and run the trainer. It may also be desirable to divide the positive and negative sets into training and test sets.

```
time python3 ./binclass.030.py --train-negative Negative_Slices_150_2/ --train-positive Positive_Slices_150_2/ --model-output handwriting_resnet_new.bin
```

