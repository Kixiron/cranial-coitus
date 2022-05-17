These two programs measure the "real" size of the array in some
sense in cells left and right of the initial cell respectively
They output the result in unary; the easiest thing is to direct them to a file
and measure its size or (on Unix) pipe the output to wc
If bounds checking is present and working the left should measure 0 and the right
should be the array size minus one

+[<+++++++++++++++++++++++++++++++++.]

+[>+++++++++++++++++++++++++++++++++.]
