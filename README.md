<!--
  ** File Name: README.md
  ** Author:    Aditya Ramesh
  ** Date:      04/27/2015
  ** Contact:   _@adityaramesh.com
-->

# Overview

- The solution for Q1 is implemented in `nngraph/nngraph_handin.lua`.
- The report file has the path `report/out/ar2922.pdf`.
- The program for sequence generation is implemented in
`lstm/a4_communication_loop.lua`. To run it, type the following commands. (Note:
due to space constraints, the word-level model is not included in this
repository.)

	cd lstm
	th a4_communication_loop.lua

- The program for character-level prediction is implemented in
`lstm/a4_char_model_loop.lua`. To run it independently of the grading script,
type the following commands:
	
	cd lstm
	th a4_char_model_loop.lua

- Lastly, the grading script has the path `lstm/a4_grading.py`. To run it, type
the following commands. Note that the interpreter must support Python 2.

	cd lstm
	python a4_grading.py
