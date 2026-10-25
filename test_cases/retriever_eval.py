
QUESTIONS = [
    {
        "id": 1,
        "question": "According to Géron, what are the main situations where machine learning is preferable to a traditional rule-based approach?",
        "ideal_answer": "Machine learning is particularly useful when traditional solutions require extensive fine-tuning or long lists of rules, when problems are too complex for traditional approaches, when the environment changes and the system needs retraining, and when we want to discover useful insights from large amounts of data.",
    },
    {
        "id": 2,
        "question": "What is the difference between supervised, unsupervised, and self-supervised learning according to the book?",
        "ideal_answer": "Supervised learning uses labeled training data. Unsupervised learning works with unlabeled data and tries to discover structure such as clusters. Self-supervised learning starts with unlabeled data but automatically generates labels—for example, masking part of an image and training the model to reconstruct it.",
    },
    {
        "id": 3,
        "question": "Why does Géron distinguish anomaly detection from novelty detection?",
        "ideal_answer": "Novelty detection assumes the training set is clean and contains no examples of the anomalies we want to detect. Anomaly detection does not make this assumption, so anomalous instances may already be present in the training data.",
    },
    {
        "id": 4,
        "question": "What are the eight major steps in the end-to-end machine learning project presented in Chapter 2?",
        "ideal_answer": "Look at the big picture; get the data; explore and visualize it; prepare the data; select and train a model; fine-tune the model; present the solution; and launch, monitor, and maintain the system.",
    },
    {
        "id": 5,
        "question": "Why does the book emphasize creating a test set early rather than repeatedly looking at the entire dataset while developing the model?",
        "ideal_answer": "Looking at the test set while developing the system introduces data snooping bias: information about the test set can influence model or feature choices, making the final evaluation overly optimistic. The test set should be kept separate so it can provide an unbiased estimate of generalization.",
    },
    {
        "id": 6,
        "question": "If a binary classifier has a very rare positive class and false positives are particularly costly, which evaluation curve does Géron recommend and why?",
        "ideal_answer": "Prefer the precision/recall (PR) curve. When the positive class is rare or false positives matter more than false negatives, the PR curve gives a more informative picture than the ROC curve.",
    },
    {
        "id": 7,
        "question": "Why can feature scaling matter for Support Vector Machines, and what does the book recommend doing?",
        "ideal_answer": "SVMs are sensitive to feature scales because the geometry of the maximum-margin decision boundary depends on the feature scales. Standardizing the features, for example with StandardScaler, can produce a much better decision boundary.",
    },
    {
        "id": 8,
        "question": "What is the fundamental idea behind large-margin classification in an SVM?",
        "ideal_answer": "The SVM chooses a decision boundary that separates the classes while keeping the largest possible margin—the widest possible street—between the classes. The training instances lying on the edge of this margin are the support vectors and determine the boundary.",
    },
    {
        "id": 9,
        "question": "Why can an unconstrained decision tree overfit, and which hyperparameters can be used to regularize it?",
        "ideal_answer": "Decision trees make few assumptions and can adapt very closely to the training data. To reduce overfitting, restrict the tree's freedom—for example with max_depth, max_leaf_nodes, min_samples_split, min_samples_leaf, or max_features. Increasing min_* parameters or decreasing max_* parameters provides regularization.",
    },
    {
        "id": 10,
        "question": "How does the book compare Gini impurity and entropy when training decision trees?",
        "ideal_answer": "They generally produce similar trees. Gini impurity is slightly faster to calculate and is therefore a good default. When they differ, Gini tends to isolate the most frequent class in its own branch, whereas entropy tends to produce somewhat more balanced trees.",
    },
    {
        "id": 11,
        "question": "What extra randomness does a Random Forest introduce compared with an ordinary decision tree, and what is the effect?",
        "ideal_answer": "At each split, a random forest considers only a random subset of the available features rather than searching all features. This increases tree diversity, trading a little more bias for lower variance and often producing a better overall model.",
    },
    {
        "id": 12,
        "question": "How are Extra-Trees made even more random than Random Forests, and what advantage does this provide?",
        "ideal_answer": "Extra-Trees additionally choose random thresholds for each candidate feature instead of searching for the optimal threshold. This increases randomness, generally trading more bias for lower variance, and makes training faster because finding optimal thresholds is computationally expensive.",
    },
    {
        "id": 13,
        "question": "In gradient boosting, what happens when the learning rate is reduced? What trade-off does this create?",
        "ideal_answer": "A lower learning rate reduces each tree's contribution. Consequently, more trees are required to fit the training data, but the resulting model generally generalizes better. Géron calls this regularization technique shrinkage.",
    },
    {
        "id": 14,
        "question": "Why might you use PCA before training another machine learning algorithm, and what does PCA sacrifice?",
        "ideal_answer": "PCA can reduce dimensionality, making training faster and reducing disk and memory requirements; in some cases it can also improve performance. The trade-off is that some information is lost. The book describes reconstruction error as the mean squared distance between the original data and the data reconstructed after dimensionality reduction.",
    },
    {
        "id": 15,
        "question": "How does a Gaussian Mixture Model differ from k-means in the way it assigns instances to clusters?",
        "ideal_answer": "K-means uses hard assignments: an instance belongs to one cluster. A Gaussian mixture model uses soft assignments, estimating the probability that an instance belongs to each cluster. During EM, these probabilities are called the clusters' responsibilities for the instances.",
    },
    {
        "id": 16,
        "question": "If you want to use a Gaussian Mixture Model for anomaly detection, how does the book suggest deciding which instances are anomalies?",
        "ideal_answer": "Use the model's estimated density. Instances in sufficiently low-density regions can be treated as anomalies. The density threshold can be selected according to the expected anomaly rate; for example, if approximately 2% of products are defective, choose a threshold that flags roughly the lowest-density 2% as anomalies.",
    },
    {
        "id": 17,
        "question": "Why can simple RNNs struggle with long sequences, and what techniques does the book discuss for addressing the unstable-gradient problem?",
        "ideal_answer": "Long sequences make the unrolled RNN very deep, which can lead to unstable gradients and very long training times. RNNs can also gradually forget earlier inputs. Possible techniques include better initialization, faster optimizers, dropout, a smaller learning rate, saturating activations such as tanh, and gradient clipping.",
    },
    {
        "id": 18,
        "question": "What problem does batch normalization help address in deep neural networks, and how does the book describe its operation?",
        "ideal_answer": "Batch normalization helps stabilize training and address unstable gradients. It normalizes activations using batch statistics and then learns scale and shift parameters. The running averages are updated using an exponential moving average controlled by the momentum parameter.",
    },
    {
        "id": 19,
        "question": "Why are positional encodings needed in a Transformer, and how does the original Transformer obtain them?",
        "ideal_answer": "Attention by itself does not inherently encode the order of tokens, so positional information must be added to the word embeddings. The original Transformer uses fixed sine/cosine positional encodings. They provide unique representations for absolute positions and also allow the model to learn relative positions.",
    },
    {
        "id": 20,
        "question": "In scaled dot-product attention, what are Q, K, and V, and why is the dot product divided by the square root of the key dimension?",
        "ideal_answer": "Q contains the queries, K the keys, and V the values. Attention is computed as softmax(QKᵀ / √dₖ)V. Dividing by √dₖ scales down the similarity scores so the softmax does not saturate, which helps avoid extremely small gradients.",
    },
]
