import numpy as np

def viterbi_nlp(words, tags, start_p, trans_p, emit_p):
    T = len(words)
    N = len(tags)
    
    # Initialize matrices (using -inf for log-space)
    viterbi_matrix = np.full((N, T), -float('inf'))
    backpointer = np.zeros((N, T), dtype=int)

    # 1. INITIALIZATION:
    # V[s,0]=log P(s)+log P(w_0∣s)
    for s in range(N):
        tag = tags[s]
        if words[0] in emit_p[tag]:
            viterbi_matrix[s, 0] = np.log(start_p[tag]) + np.log(emit_p[tag][words[0]])

    # 2. RECURSION:
    # V[j,t]= max_{i=1}^N (V[i,t−1]+log P(t_j​ ∣t_i​ )+log P(w_t​ ∣ t_j​ ))
    for t in range(1, T):
        for j in range(N): # Current tag
            curr_tag = tags[j]
            for i in range(N): # Previous tag
                prev_tag = tags[i]
                
                # Calculate path probability
                if words[t] in emit_p[curr_tag]:
                    score = viterbi_matrix[i, t-1] + \
                            np.log(trans_p[prev_tag][curr_tag]) + \
                            np.log(emit_p[curr_tag][words[t]])
                    
                    if score > viterbi_matrix[j, t]:
                        viterbi_matrix[j, t] = score
                        backpointer[j, t] = i

    # 3. TERMINATION: Find best ending
    best_last_tag_idx = np.argmax(viterbi_matrix[:, T-1])
    
    # 4. BACKTRACKING: Trace the path
    best_path_indices = [best_last_tag_idx]
    for t in range(T-1, 0, -1):
        best_last_tag_idx = backpointer[best_last_tag_idx, t]
        best_path_indices.append(best_last_tag_idx)
        
    return [tags[i] for i in reversed(best_path_indices)]


words = ["Janet", "will", "back", "the", "bill"]

tags = ["NNP", "MD", "VB", "JJ", "NN", "RB", "DT"]

start_p = {
    "NNP": 0.2767, "MD": 0.0006, "VB": 0.0031, "JJ": 0.0453, 
    "NN": 0.0449, "RB": 0.0510, "DT": 0.2026
}

trans_p = {
    "NNP": {"NNP": 0.3777, "MD": 0.0110, "VB": 0.0009, "JJ": 0.0084, "NN": 0.0584, "RB": 0.0090, "DT": 0.0025},
    "MD":  {"NNP": 0.0008, "MD": 0.0002, "VB": 0.7968, "JJ": 0.0005, "NN": 0.0008, "RB": 0.1698, "DT": 0.0041},
    "VB":  {"NNP": 0.0322, "MD": 0.0005, "VB": 0.0050, "JJ": 0.0837, "NN": 0.0615, "RB": 0.0514, "DT": 0.2231},
    "JJ":  {"NNP": 0.0366, "MD": 0.0004, "VB": 0.0001, "JJ": 0.0733, "NN": 0.4509, "RB": 0.0036, "DT": 0.0036},
    "NN":  {"NNP": 0.0096, "MD": 0.0176, "VB": 0.0014, "JJ": 0.0086, "NN": 0.1216, "RB": 0.0177, "DT": 0.0068},
    "RB":  {"NNP": 0.0068, "MD": 0.0102, "VB": 0.1011, "JJ": 0.1012, "NN": 0.0120, "RB": 0.0728, "DT": 0.0479},
    "DT":  {"NNP": 0.1147, "MD": 0.0021, "VB": 0.0002, "JJ": 0.2157, "NN": 0.4744, "RB": 0.0102, "DT": 0.0017}
}

emit_p = {
    "NNP": {"Janet": 0.000032},
    "MD":  {"will": 0.308431},
    "VB":  {"will": 0.000028, "back": 0.000672, "bill": 0.000028},
    "JJ":  {"back": 0.000340},
    "NN":  {"will": 0.000200, "back": 0.000223, "bill": 0.002337},
    "RB":  {"back": 0.010446},
    "DT":  {"the": 0.506099}
}

print(viterbi_nlp(words, tags, start_p, trans_p, emit_p))
