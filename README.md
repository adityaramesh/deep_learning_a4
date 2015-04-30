<!--
  ** File Name: README.md
  ** Author:    Aditya Ramesh
  ** Date:      04/29/2015
  ** Contact:   _@adityaramesh.com
-->

# Important Note

You will need to download the character-level model file from the following URL,
and save it to the path `models/queryable_char_model/best_train_model.t7`.  I
could not include it in the repository because the file size is over 100 Mb.
Downloading the file cannot be done normally using `wget`, so I could not
automate this process.

	https://github.com/adityaramesh/a4/releases/download/models/best_train_model.t7

# Overview

- The solution for Q1 is implemented in `nngraph/nngraph_handin.lua`.
- The report file has the path `report/out/ar2922.pdf`.
- The program for sequence generation is implemented in
`a4_communication_loop.lua`. To run it, type `th
a4_communication_loop.lua`. (Note: due to space constraints, the word-level
model is not included in this repository.)
- The program for character-level prediction is implemented in
`a4_char_model_loop.lua`. To run it independently of the grading script,
type `th a4_char_model_loop.lua`.
- Lastly, the grading script has the path `a4_grading.py`. To run it, type
the `python a4_grading.py`. Note that the interpreter must support Python 2.
