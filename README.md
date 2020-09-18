# Darks and Stripes: Effects of Clothing on Weight Perception

Code for the publication

> Kirill Martynov, Kiran Garimella, Robert West: **Darks and Stripes: Effects of Clothing on Weight Perception.** *Journal of Social Computing,* 2020.

This repo contains the code that we used to collect, filter, annotate, and analyze the data as described in the article.

Note that the paths to our data files are hardcoded in multiple locations. The data files are available upon request from the authors.

Repository structure:

* Pre-processing of raw MTurk results:
    * parse_results.py

* Main analysis:
    * analyse_pairs.ipynb
    * process_pairs.py
    * process_tc.py

* Annotation:
    * matching.py
    * color_detection/

* Automation of tasks assocaited with carrying out experiments on MTurk:
    * compute_cost.py
    * create_mturk_paired_tasks.py
    * reject_assignments.py
    * update_qualification.py

* Minor analysis:
    * compare_pair.py
    * plot_time_evolution.py
    * plot_workers.py

* Minor helpers:
    * routines.py
    * split_attribute.py
    * split_images_color.py
