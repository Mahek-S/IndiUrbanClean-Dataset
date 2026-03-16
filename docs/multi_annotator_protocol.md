# Multi-Annotator Protocol

To maintain reliable and consistent labeling in the IndiUrbanClean dataset, a multi-annotator protocol was followed during the annotation process. Since some urban waste scenes can visually overlap across categories, multiple annotators and a verification step were used to improve labeling quality.

## Step 1: Independent Annotation

Two annotators independently labeled the same 100 randomly selected images from the dataset.
Each annotator assigned one of the five class labels based on the visible waste condition in the image.

## Step 2: Label Comparison

The annotations from Annotator_A and Annotator_B were compared to identify cases of agreement and disagreement.
Images with matching labels were accepted as correctly labeled.

## Step 3: Review of Disagreements

Images with label disagreements were manually reviewed and discussed by the annotators.
Most disagreements occurred between visually similar categories such as:

* open_waste vs dumpyard
* open_waste vs overfilled_dustbins
* construction_waste vs open_waste

These cases were analyzed to better understand the boundaries between the classes.

## Step 4: Refinement of Annotation Guidelines

Based on the discussion of disagreements, the annotation guidelines were refined to make class boundaries clearer. Additional rules were defined for situations where multiple waste conditions appeared in the same image.

## Step 5: Handling Edge Cases

Specific rules were applied to resolve ambiguous cases observed during annotation:

**Open waste vs Overfilled bins**
If a dustbin was present and waste was visibly overflowing from the container, the image was labeled as **overfilled_bins**, even if some waste was scattered on the ground nearby.
However, if most of the waste appeared on the ground and was not clearly associated with the dustbin, the image was labeled as **open_waste**.

**Open waste vs Dumpyard**
Waste scattered along roads, sidewalks, or roadside areas was labeled as **open_waste**.
Images showing large concentrated garbage piles in a specific dumping location (such as municipal dumping areas or large garbage heaps) were labeled as **dumpyard**.

**Open waste vs Construction waste**
If the waste mainly consisted of construction materials such as bricks, concrete debris, sand piles, or building materials, the image was labeled as **construction_waste**.
If construction debris appeared together with general garbage and the overall scene resembled scattered waste, the image was labeled as **open_waste**.

## Step 6: Final Annotation

After refining the annotation rules, the remaining images in the dataset were annotated following the updated guidelines. This process helped improve labeling consistency and reduce ambiguity across the dataset.
