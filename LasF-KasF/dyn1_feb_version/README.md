
--- Basic usage ---
Simply use command "python3 main.py" to train a DyN1-hybrid language model from scratch on corpis such as WikiText103 and PennTreebank.

--- Some arguments/variables in the script ---
bptt: backpropagtion through time, sequence length during training.
emsize: embedding dimension of the input encodings.
dyn_Hdim: hidden dimension of a dyn blocl
p_norm: use Lp-norm as the metric function to measure the relation between neuronal states.
q_dim: the dimension of neuronal states.
num_D: the number of sub-columns per cortical column.

--- cortical_column ---
The fundamental units that process the input signals to generate output signals and the current neuronal states.
Each cortical column refers to a neuronal community.

--- neuronal_broadcast ---
This block involves several neuronal broadcasting/communicating mechanisms that enable each neural community to determine how to integrate the global signals based on the global dynamics, resulting in the ultimate output. 

--- dyn_LM ---
We use typical self-regression mechanism here to build a dyn_LM by replacing the typical Transformer blocks with our proposed coritical column (neuronal community) whose signal generation mechanism follows the neuronal broadcasting block.

--- train_utils ---
We evaluate two kinds of PPL(perplexity) for language modeling task.
The first PPL is the typical one.
The second PPL denoted as lPPL measures the next-token prediction capacity for the next of the last token of a sequence.
This token is unseen amongst the input tokens.
The lPPL is a self-designed indicator that might be relevant to the generalization capacity of a LM.


