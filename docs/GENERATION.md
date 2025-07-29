# GIFARC pipeline

To generate ARC-style tasks with GIFARC pipeline...

Open, description_executor.ipynb and now run all the ipynb file, it will generate the gif arc under ./results/

You need to use run all to generate data continuously however if it failed also want to use prev data then make sure the folder name that you want to use.

# Error 

Because of Prompt Parsing issue and code from llm could be wrong, it sometime make an error with generating code which hard to check error, when the data is failed to generate check metadata and see the column of result is 0(failed) or 1(successed).
Sometimes even it show 1 might file and missing or empty

