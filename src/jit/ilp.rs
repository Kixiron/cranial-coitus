// https://grothoff.org/christian/lcpc2006.pdf
//
// (1) Cost of spills - i is the temporary Si is the cost of that spill, xi,o is a boolean
//     true or false if that temporary was spilled or not. Its saying the aim is to lower this cost
// (2) A constraint saying that the number of bits required does not exceed the bits available (b) in the register
// (3) and (4) All temporaries must be assigned to exactly one location
//
// Inputs:
// i    - The set of temporaries
// r    - The set of registers available
// Si   - The cost to spill the temporary i to the stack
// wi,n - The width of the temporary i in bits, at a given CFG node n (You will most likely only need zero
//        (meaning not needed at this node) and wordsize)
//
// Output is Xi,r - i is the temporary, r is the register. Given the value {0,1} to show where it was placed
