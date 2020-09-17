# darks_and_stripes
Code for publication 'Darks and Stripes: Effects of Clothing on Weight Perception'

This repo contains the code that we used to collect, filter, annotate, and analyse the data as described in the article. The code is explorative and was written without any intent to maintain or reuse it in the future. This is just a fancy way to say that large pieces of the code may appear consufing to a reader outside of our research group. If you struggle to understand it and need clarifications, please reach out to the authors. 

Note that the paths to our data files are hardcoded at multiple locations. To be on the safe side with regards to privacy, we chose not to share the openly and make it available only upon request.

Below we briefly describe the code structure:

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
