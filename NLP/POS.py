import numpy as np
from collections import defaultdict

class ViterbiTagger:
    def __init__(self):
        self.tags = []
        self.start_p = {}
        self.trans_p = defaultdict(lambda: defaultdict(float))
        self.emit_p = defaultdict(lambda: defaultdict(float))

    def train(self, corpus):
        """
        corpus: List of list of tuples [(word, tag), ...]
        """
        tag_counts = defaultdict(int)
        transition_counts = defaultdict(lambda: defaultdict(int))
        emission_counts = defaultdict(lambda: defaultdict(int))
        start_counts = defaultdict(int)

        for sentence in corpus:
            for i, (word, tag) in enumerate(sentence):
                tag_counts[tag] += 1
                emission_counts[tag][word] += 1
                
                if i == 0:
                    start_counts[tag] += 1
                else:
                    prev_tag = sentence[i-1][1]
                    transition_counts[prev_tag][tag] += 1

        self.tags = list(tag_counts.keys())
        total_sentences = len(corpus)

        # Convert counts to Log-Probabilities to avoid underflow
        for tag in self.tags:
            # Start Probabilities
            self.start_p[tag] = np.log((start_counts[tag] + 1) / (total_sentences + len(self.tags)))
            
            # Transition Probabilities
            for next_tag in self.tags:
                count = transition_counts[tag][next_tag]
                self.trans_p[tag][next_tag] = np.log((count + 1) / (tag_counts[tag] + len(self.tags)))
            
            # Emission Probabilities
            for word, count in emission_counts[tag].items():
                self.emit_p[tag][word] = np.log((count + 1) / (tag_counts[tag] + len(emission_counts[tag])))

    def tag(self, sentence):
        T = len(sentence)
        N = len(self.tags)
        
        viterbi_matrix = np.full((N, T), -float('inf'))
        backpointer = np.zeros((N, T), dtype=int)

        # Initialization
        for s in range(N):
            tag = self.tags[s]
            # Use a small default log value for unknown words
            emission = self.emit_p[tag].get(sentence[0], -20.0) 
            viterbi_matrix[s, 0] = self.start_p[tag] + emission

        # Recursion
        for t in range(1, T):
            for j in range(N):
                curr_tag = self.tags[j]
                emission = self.emit_p[curr_tag].get(sentence[t], -20.0)
                
                # Vectorized search for max probability from previous states
                scores = viterbi_matrix[:, t-1] + \
                         np.array([self.trans_p[self.tags[i]][curr_tag] for i in range(N)]) + \
                         emission
                
                viterbi_matrix[j, t] = np.max(scores)
                backpointer[j, t] = np.argmax(scores)

        # Backtracking
        best_path_idx = [np.argmax(viterbi_matrix[:, T-1])]
        for t in range(T-1, 0, -1):
            best_path_idx.append(backpointer[best_path_idx[-1], t])
            
        return [self.tags[i] for i in reversed(best_path_idx)]

# --- EXAMPLE USAGE ---


# 2. Initialize and Train
tagger = ViterbiTagger()
tagger.train(train_data)

# 3. Test on a new sentence
test_sentence = ["the", "bat", "flies"]
predicted_tags = tagger.tag(test_sentence)

print(f"Sentence: {test_sentence}")
print(f"Tags:     {predicted_tags}")
