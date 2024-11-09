# Project Started

## Start to Install requierments


- install requierments libraries
- Set Git  & Github
- Create a Basic Project Structure
- Install & Create conda env to start learning and  testing

## Start to Learn Concepts

- Learning Transformer Base Architecture
- Learning Self-Attention
- Learning Tokenization
- Learning Vectorization ((TF-IDF, Word2Vec))
- Learning Embedding
- Learning Transformer Models (BERT/RoBERTa)


# **Understanding Transformers and Self-Attention Mechanisms: A Comprehensive Training Guide**



## **Table of Contents**

1. [Introduction to Transformers](#1-introduction-to-transformers)
2. [Word Embeddings](#2-word-embeddings)
3. [Positional Encoding](#3-positional-encoding)
4. [Self-Attention Mechanism](#4-self-attention-mechanism)
    - [4.1. Overview](#41-overview)
    - [4.2. Query, Key, and Value Matrices](#42-query-key-and-value-matrices)
    - [4.3. Calculating Attention Scores](#43-calculating-attention-scores)
    - [4.4. Applying Softmax](#44-applying-softmax)
    - [4.5. Generating the Output](#45-generating-the-output)
5. [Numerical Example of Self-Attention](#5-numerical-example-of-self-attention)
    - [5.1. Step 1: Define the Input Sentence and Embeddings](#51-step-1-define-the-input-sentence-and-embeddings)
    - [5.2. Step 2: Define Query, Key, and Value Matrices](#52-step-2-define-query-key-and-value-matrices)
    - [5.3. Step 3: Calculate Query, Key, and Value Vectors](#53-step-3-calculate-query-key-and-value-vectors)
    - [5.4. Step 4: Calculate Attention Scores](#54-step-4-calculate-attention-scores)
    - [5.5. Step 5: Compute the Output](#55-step-5-compute-the-output)
6. [Detailed Explanation of the Output](#6-detailed-explanation-of-the-output)
7. [Final Thoughts](#7-final-thoughts)
8. [Glossary](#8-glossary)


---

## **1. Introduction to Transformers**

Transformers have revolutionized the field of natural language processing (NLP) by enabling models to handle long-range dependencies in data more effectively than previous architectures like recurrent neural networks (RNNs) or convolutional neural networks (CNNs). Introduced in the paper ["Attention Is All You Need"](/Docs/references/NIPS-2017-attention-is-all-you-need-Paper.pdf) by Vaswani et al., Transformers leverage a mechanism known as **self-attention** to process input data in parallel, making them highly efficient and scalable.

### **Key Components of Transformers:**

- **Encoder-Decoder Structure:** Transformers typically consist of an encoder that processes the input and a decoder that generates the output.
- **Self-Attention Mechanism:** Allows the model to weigh the importance of different parts of the input data dynamically.
- **Multi-Head Attention:** Enhances the model's ability to focus on different representation subspaces.
- **Positional Encoding:** Introduces information about the order of the sequence since Transformers lack inherent sequential processing.

This guide focuses on the **self-attention mechanism**, a core component of Transformers, detailing how it operates and providing a numerical example to illustrate its functionality.

---

## **2. Word Embeddings**

Before delving into self-attention, it's essential to understand how words are represented numerically within Transformer models.

### **What Are Word Embeddings?**

Word embeddings are dense vector representations of words in a continuous vector space where semantically similar words are mapped to nearby points. They capture syntactic and semantic information about words, enabling models to understand relationships and contexts.

### **Common Methods for Generating Embeddings:**

- **Pre-trained Embeddings:** Models like Word2Vec, GloVe, and FastText provide pre-trained embeddings based on large corpora.
- **Learned Embeddings:** In Transformer architectures, embeddings are often learned during the training process, allowing them to be fine-tuned for specific tasks.

### **Example:**

Consider the sentence: **"I love AI"**

Each word is mapped to a **d-dimensional** vector. For simplicity, we'll use 3-dimensional vectors in this example:

- **I** → [1, 0, 1]
- **love** → [0, 1, 0]
- **AI** → [1, 1, 0]

Thus, the input embedding matrix $ X $ (3 words × 3 dimensions) is:

$$
X = 
\begin{bmatrix}
1 & 0 & 1 \\
0 & 1 & 0 \\
1 & 1 & 0
\end{bmatrix}
$$

---

## **3. Positional Encoding**

Transformers process input data in parallel, lacking inherent sequential information. To inject information about the order of words in a sequence, **positional encodings** are added to the word embeddings.

### **Why Positional Encoding?**

Without positional encoding, the model treats the input as a bag of words, ignoring the order, which is crucial for understanding context and meaning.

### **How It Works:**

Positional encodings are vectors added to the word embeddings, encoding the position of each word in the sequence. These encodings can be:

- **Fixed:** Using sine and cosine functions of different frequencies.
- **Learned:** Parameters learned during training.

### **Example:**

Continuing with our sentence **"I love AI"**, suppose we add simple positional encodings:

- **I** (position 1) → [1, 0, 1] + [0.1, 0.2, 0.3] = [1.1, 0.2, 1.3]
- **love** (position 2) → [0, 1, 0] + [0.2, 0.3, 0.4] = [0.2, 1.3, 0.4]
- **AI** (position 3) → [1, 1, 0] + [0.3, 0.4, 0.5] = [1.3, 1.4, 0.5]

The updated embedding matrix with positional encodings becomes:

$$
\text{Embedding matrix} =
\begin{bmatrix}
1.1 & 0.2 & 1.3 \\
0.2 & 1.3 & 0.4 \\
1.3 & 1.4 & 0.5
\end{bmatrix}
$$

---

## **4. Self-Attention Mechanism**

Self-attention allows the model to weigh the relevance of different words in a sequence relative to each other, enabling the model to capture dependencies irrespective of their distance in the sequence.

### **4.1. Overview**

In the self-attention mechanism:

- **Each word** in the input sequence is transformed into three vectors: **Query (Q)**, **Key (K)**, and **Value (V)**.
- **Attention scores** are calculated using the dot product of Query and Key vectors.
- These scores determine how much attention each word should pay to others.
- The final output is a weighted sum of the Value vectors based on these attention scores.

### **4.2. Query, Key, and Value Matrices**

**Query (Q):** Represents the word seeking information.

**Key (K):** Represents the word providing information.

**Value (V):** Contains the actual information to be aggregated.

These matrices are obtained by applying learned linear transformations to the input embeddings.

### **4.3. Calculating Attention Scores**

Attention scores are computed by taking the dot product of the Query vector of one word with the Key vectors of all words, including itself. These scores indicate the relevance of each word to the Query word.

### **4.4. Applying Softmax**

To convert raw attention scores into probabilities, the **softmax** function is applied. This ensures that the attention weights sum to 1, allowing them to be interpreted as probabilities.

### **4.5. Generating the Output**

The final output for each word is a weighted sum of the Value vectors, where the weights are the attention probabilities. This output incorporates contextual information from the entire sequence.

---

## **5. Numerical Example of Self-Attention**

To elucidate the self-attention mechanism, let's walk through a detailed numerical example using the sentence **"I love AI"**.

### **5.1. Step 1: Define the Input Sentence and Embeddings**

Assume the sentence: **"I love AI"**.

Each word is assigned a 3-dimensional embedding vector:

- **I** → [1, 0, 1]
- **love** → [0, 1, 0]
- **AI** → [1, 1, 0]

The input embedding matrix $ X $ is:

$$
X = 
\begin{bmatrix}
1 & 0 & 1 \\
0 & 1 & 0 \\
1 & 1 & 0
\end{bmatrix}
$$

### **5.2. Step 2: Define Query, Key, and Value Matrices**

Define the weight matrices $ W_Q $, $ W_K $, and $ W_V $ (all 3×3 for simplicity):

$$
W_Q = 
\begin{bmatrix}
1 & 0 & 1 \\
0 & 1 & 0 \\
1 & 0 & 1
\end{bmatrix}
,\quad
W_K = 
\begin{bmatrix}
1 & 1 & 0 \\
0 & 1 & 1 \\
1 & 0 & 1
\end{bmatrix}
,\quad
W_V = 
\begin{bmatrix}
1 & 0 & 1 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

These matrices are **learned** during the training process and are used to project the input embeddings into the Query, Key, and Value spaces.

### **5.3. Step 3: Calculate Query, Key, and Value Vectors**

Compute $ Q $, $ K $, and $ V $ by multiplying the input embeddings $ X $ with the respective weight matrices.

#### **Query Matrix (Q):**

$$
Q = X \times W_Q = 
\begin{bmatrix}
1 & 0 & 1 \\
0 & 1 & 0 \\
1 & 1 & 0
\end{bmatrix}
\times
\begin{bmatrix}
1 & 0 & 1 \\
0 & 1 & 0 \\
1 & 0 & 1
\end{bmatrix}
=
\begin{bmatrix}
(1\cdot1 + 0\cdot0 + 1\cdot1) & (1\cdot0 + 0\cdot1 + 1\cdot0) & (1\cdot1 + 0\cdot0 + 1\cdot1) \\
(0\cdot1 + 1\cdot0 + 0\cdot1) & (0\cdot0 + 1\cdot1 + 0\cdot0) & (0\cdot1 + 1\cdot0 + 0\cdot1) \\
(1\cdot1 + 1\cdot0 + 0\cdot1) & (1\cdot0 + 1\cdot1 + 0\cdot0) & (1\cdot1 + 1\cdot0 + 0\cdot1)
\end{bmatrix}
=
\begin{bmatrix}
2 & 0 & 2 \\
0 & 1 & 0 \\
1 & 1 & 1
\end{bmatrix}
$$

#### **Key Matrix (K):**

$$
K = X \times W_K = 
\begin{bmatrix}
1 & 0 & 1 \\
0 & 1 & 0 \\
1 & 1 & 0
\end{bmatrix}
\times
\begin{bmatrix}
1 & 1 & 0 \\
0 & 1 & 1 \\
1 & 0 & 1
\end{bmatrix}
=
\begin{bmatrix}
(1\cdot1 + 0\cdot0 + 1\cdot1) & (1\cdot1 + 0\cdot1 + 1\cdot0) & (1\cdot0 + 0\cdot1 + 1\cdot1) \\
(0\cdot1 + 1\cdot0 + 0\cdot1) & (0\cdot1 + 1\cdot1 + 0\cdot0) & (0\cdot0 + 1\cdot1 + 0\cdot1) \\
(1\cdot1 + 1\cdot0 + 0\cdot1) & (1\cdot1 + 1\cdot1 + 0\cdot0) & (1\cdot0 + 1\cdot1 + 0\cdot1)
\end{bmatrix}
=
\begin{bmatrix}
2 & 1 & 1 \\
0 & 1 & 1 \\
1 & 2 & 1
\end{bmatrix}
$$

#### **Value Matrix (V):**

$$
V = X \times W_V = 
\begin{bmatrix}
1 & 0 & 1 \\
0 & 1 & 0 \\
1 & 1 & 0
\end{bmatrix}
\times
\begin{bmatrix}
1 & 0 & 1 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
=
\begin{bmatrix}
(1\cdot1 + 0\cdot0 + 1\cdot0) & (1\cdot0 + 0\cdot1 + 1\cdot0) & (1\cdot1 + 0\cdot0 + 1\cdot1) \\
(0\cdot1 + 1\cdot0 + 0\cdot0) & (0\cdot0 + 1\cdot1 + 0\cdot0) & (0\cdot1 + 1\cdot0 + 0\cdot1) \\
(1\cdot1 + 1\cdot0 + 0\cdot0) & (1\cdot0 + 1\cdot1 + 0\cdot0) & (1\cdot1 + 1\cdot0 + 0\cdot1)
\end{bmatrix}
=
\begin{bmatrix}
1 & 0 & 2 \\
0 & 1 & 0 \\
1 & 1 & 1
\end{bmatrix}
$$

### **5.4. Step 4: Calculate Attention Scores**

Attention scores determine how much focus each word should have on others. They are computed using the dot product of the Query matrix $ Q $ with the transpose of the Key matrix $ K^T $, scaled by the square root of the key dimension $ d_k $.

#### **Formula:**

$$
\text{Attention scores} = \frac{Q \times K^T}{\sqrt{d_k}}
$$

Given $ d_k = 3 $, $ \sqrt{d_k} \approx 1.732 $.

#### **Calculating** $ Q \times K^T $:

$$
K^T = 
\begin{bmatrix}
2 & 0 & 1 \\
1 & 1 & 2 \\
1 & 1 & 1
\end{bmatrix}
,\quad
Q \times K^T =
\begin{bmatrix}
2 & 0 & 2 \\
0 & 1 & 0 \\
1 & 1 & 1
\end{bmatrix}
\times
\begin{bmatrix}
2 & 0 & 1 \\
1 & 1 & 2 \\
1 & 1 & 1
\end{bmatrix}
=
\begin{bmatrix}
(2\cdot2 + 0\cdot1 + 2\cdot1) & (2\cdot0 + 0\cdot1 + 2\cdot1) & (2\cdot1 + 0\cdot2 + 2\cdot1) \\
(0\cdot2 + 1\cdot1 + 0\cdot1) & (0\cdot0 + 1\cdot1 + 0\cdot1) & (0\cdot1 + 1\cdot2 + 0\cdot1) \\
(1\cdot2 + 1\cdot1 + 1\cdot1) & (1\cdot0 + 1\cdot1 + 1\cdot1) & (1\cdot1 + 1\cdot2 + 1\cdot1)
\end{bmatrix}
=
\begin{bmatrix}
6 & 2 & 4 \\
1 & 1 & 2 \\
4 & 2 & 3
\end{bmatrix}
$$

#### **Scaling by  \sqrt{d_k} :**

$$
\frac{1}{\sqrt{3}} \times
\begin{bmatrix}
6 & 2 & 4 \\
1 & 1 & 2 \\
4 & 2 & 3
\end{bmatrix}
=
\begin{bmatrix}
3.46 & 1.15 & 2.31 \\
0.58 & 0.58 & 1.15 \\
2.31 & 1.15 & 1.73
\end{bmatrix}
$$

### **5.5. Step 5: Compute the Output**

The final output is obtained by multiplying the attention scores (after applying softmax) with the Value matrix $ V $.

#### **Applying Softmax:**

Softmax is applied row-wise to normalize the attention scores into probabilities.

**Example: Softmax for the First Row:**

$$
\text{Softmax}(3.46, 1.15, 2.31) = \left( \frac{e^{3.46}}{e^{3.46} + e^{1.15} + e^{2.31}}, \frac{e^{1.15}}{e^{3.46} + e^{1.15} + e^{2.31}}, \frac{e^{2.31}}{e^{3.46} + e^{1.15} + e^{2.31}} \right)
$$

Assuming the softmax results for simplicity:

$$
\text{Attention scores} =
\begin{bmatrix}
0.6 & 0.1 & 0.3 \\
0.2 & 0.5 & 0.3 \\
0.3 & 0.2 & 0.5
\end{bmatrix}
$$

#### **Multiplying by Value Matrix V :**

$$
\text{Output} = \text{Attention scores} \times V =
\begin{bmatrix}
0.6 & 0.1 & 0.3 \\
0.2 & 0.5 & 0.3 \\
0.3 & 0.2 & 0.5
\end{bmatrix}
\times
\begin{bmatrix}
1 & 0 & 2 \\
0 & 1 & 0 \\
1 & 1 & 1
\end{bmatrix}
=
\begin{bmatrix}
(0.6\cdot1 + 0.1\cdot0 + 0.3\cdot1) & (0.6\cdot0 + 0.1\cdot1 + 0.3\cdot1) & (0.6\cdot2 + 0.1\cdot0 + 0.3\cdot1) \\
(0.2\cdot1 + 0.5\cdot0 + 0.3\cdot1) & (0.2\cdot0 + 0.5\cdot1 + 0.3\cdot1) & (0.2\cdot2 + 0.5\cdot0 + 0.3\cdot1) \\
(0.3\cdot1 + 0.2\cdot0 + 0.5\cdot1) & (0.3\cdot0 + 0.2\cdot1 + 0.5\cdot1) & (0.3\cdot2 + 0.2\cdot0 + 0.5\cdot1)
\end{bmatrix}
=
\begin{bmatrix}
1.3 & 0.4 & 1.5 \\
0.5 & 0.8 & 1.2 \\
0.8 & 0.7 & 1.5
\end{bmatrix}
$$

The **output matrix** represents the **self-attended embeddings** for each word in the sentence, incorporating contextual information from the entire sequence.

---

## **6. Detailed Explanation of the Output**

The output of the self-attention mechanism transforms each word's embedding by incorporating information from other words in the sequence, weighted by the attention scores. This process enables the model to generate **contextualized representations** of each word, which are essential for understanding nuances in language.

### **Understanding the Output Matrix:**

$$
\text{Output} = 
\begin{bmatrix}
1.3 & 0.4 & 1.5 \quad (\text{for "I"}) \\
0.5 & 0.8 & 1.2 \quad (\text{for "love"}) \\
0.8 & 0.7 & 1.5 \quad (\text{for "AI"})
\end{bmatrix}
$$

#### **Interpretation:**

- **Output for "I":** [1.3, 0.4, 1.5]
    - **0.6** of "I"'s representation comes from itself.
    - **0.1** comes from "love."
    - **0.3** comes from "AI."

- **Output for "love":** [0.5, 0.8, 1.2]
    - **0.2** from "I."
    - **0.5** from itself.
    - **0.3** from "AI."

- **Output for "AI":** [0.8, 0.7, 1.5]
    - **0.3** from "I."
    - **0.2** from "love."
    - **0.5** from itself.

### **Significance of the Output:**

- **Contextual Awareness:** Each word's embedding now contains information about other words in the sentence, enabling the model to grasp context more effectively.
- **Dynamic Weighting:** The attention mechanism dynamically adjusts the influence of each word based on relevance, allowing the model to focus on pertinent information.
- **Enhanced Representations:** These enriched embeddings serve as inputs to subsequent layers in the Transformer, facilitating deeper understanding and more accurate predictions.

---

## **7. Final Thoughts**

The self-attention mechanism is a cornerstone of Transformer architectures, empowering models to process and understand language with unprecedented efficiency and accuracy. By dynamically weighing the importance of each word in a sequence relative to others, self-attention facilitates the creation of rich, contextualized representations that are vital for a wide array of NLP tasks, including translation, summarization, and question-answering.

### **Key Takeaways:**

- **Parallel Processing:** Unlike sequential models, Transformers process all words simultaneously, enhancing computational efficiency.
- **Scalability:** The architecture scales well with large datasets and complex tasks.
- **Versatility:** Self-attention mechanisms are not limited to NLP and can be applied to other domains like computer vision.

Understanding the intricacies of self-attention and its implementation within Transformers is essential for leveraging these models effectively in various applications.

---

## **8. Glossary**

- **Attention Mechanism:** A technique that allows models to focus on specific parts of the input data when generating each part of the output.
- **Embedding Vector:** A numerical representation of a word in a continuous vector space.
- **Query (Q):** A vector representing the current word seeking information.
- **Key (K):** A vector representing a word that provides information.
- **Value (V):** A vector containing the actual information to be aggregated.
- **Softmax:** A function that converts a vector of values into probabilities that sum to 1.
- **Positional Encoding:** Information added to embeddings to encode the position of words in a sequence.
- **Multi-Head Attention:** An extension of self-attention that allows the model to focus on different representation subspaces.
- **Transformer:** A neural network architecture that relies entirely on self-attention mechanisms to process input data.

