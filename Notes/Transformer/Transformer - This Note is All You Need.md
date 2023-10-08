# How Should We Learn Transformer
As we move into Transformer, we should understand that not every part of the structure is fully understandable. Some are just sewing the historically proved useful methods together, and some are empirical decisions made by the inventors of Transformer.

In every part of Transformer structure, we should always try the best to find the boundary between what's understandable and explanable and what's not. For those we cannot, there's no need to be frustrated by them, as AI itself is too complex and mythological for human to understand.

The following analysis are all based on trained transformer model, meaning forward propagation only. We are trying to figure out from a structural perspective how a well-trained transformer is able to make predictions given an input sentence.

# Problems of RNN & Transformer Solution
1. __Problem:__ Long range dependencies. __Solution:__ Transformer uses self-attention to integrate information in all words.
2. __Problem:__ Gradient vanishing or exploding . __Solution:__ Transformer's parallel computing has very shallow network comparing to RNN
3. __Problem:__ Training steps depend on input sentence length, might be too long. __Solution:__ Transformer reads the sentence as a whole, so no iteration.
4. __Problem:__ Structure not allow parallel computation. __Solution:__ Transformer supports parallel computation.


# Self Attention

## What is Self Attention
As we know, attention uses weighted mean sum of the activations to compute the outputs, so self attention applies the same weighted mean sum to the sentence itself.\
In the simplest self attention, for every word, self attention calculates the weighted mean sum of all words in the sentence (including the word itself) based on the relevance of this word and the other words.
$$
w_{11} = \vec{v}_1^T\vec{v}_1, w_{12} = \vec{v}_1^T\vec{v}_2,..., w_{1n} = \vec{v}_1^T\vec{v}_n\\
w_{21} = \vec{v}_2^T\vec{v}_1, w_{22} = \vec{v}_2^T\vec{v}_2,..., w_{2n} = \vec{v}_2^T\vec{v}_n\\
w_{n1} = \vec{v}_n^T\vec{v}_1, w_{n2} = \vec{v}_n^T\vec{v}_2,..., w_{nn} = \vec{v}_n^T\vec{v}_n
$$
$W_ij$ are the weights, and $i$ and $j$ represents the target word and all the words compared with the target word.

With the weights, the weight mean sum works as the following
$$
\vec{y}_1 = w_{11}\vec{v}_1 + w_{12}\vec{v}_2 + \dots + w_{1n}\vec{v}_n\\
\vec{y}_2 = w_{21}\vec{v}_1 + w_{22}\vec{v}_2 + \dots + w_{2n}\vec{v}_n\\
\vec{y}_n = w_{n1}\vec{v}_1 + w_{n2}\vec{v}_2 + \dots + w_{nn}\vec{v}_n
$$
To better understand this, we see that the weights are calculated through word vector dot products, so the weight is greater if the two vectors are closer, or more relevant. As the weights calculated from $\vec{v}_i$ and $\vec{v}_j$ is again multiplied by $\vec{v}_j$, $\vec{y}_i$ naturally gets more information from $\vec{v}_j$ that are most relevant to $\vec{v}_i$, and less info from those irrelevant.

However, what does this "relevance" actually presents? How can the machine infer sentence meanings and grammars just based on the relevance? These are the questions unanswered. As we move on to improved self attention with Query, Key, and Value, this problem gets more intense.

## Query, Key, and Value (QKV)
### Why QKV
The relevance between word vectors in simple self attention is defined as the distance between word vectors. This is something we thought explanable that closer word vectors are more relevant, but the truth is, it most certainly is not the best way to describe relevance.

Thus, instead of artificially design a way to calculate the relevance (as human designs hardly cover all cases and reach the optimal method), we decide to let the machine make the decision. So we multiply weight vectors that can be trained to the word vectors and __TRUST__ the back propagation can lead us to the best relevance.

### How to Understand QKV
QKV originates from database, and it is a process to find the best-match value from a Query, several Keys and corresponding Values. For QKV in self attention, the process is almost the same.\
The difference between self attention and database is that self attention doesn't retrieve one value throught a key, instead it calculates the probability of each key being the right answer and weighted-mean sum the values behind the keys.\
__To understand the process from word meanings,__ Query as a modified word vector is compared to the Keys (word vectors modified in a different way) to calculate the weights that represent each's relevance. The Values (word vectors modified again) are then multiplied by the weights to get the final integrated information for this Query. And the Query is repeated $n$ times as there are $n$ word vectors.

The way to calculate the relevance between Query and Keys is called the similarity method. There are several methods to achieve this goal.
1. dot product: $s_{ij} = \vec{q}_i^T\vec{k}_j$, the method used for simple self attention, but the vectors are word vectors modified by trained weight matrices.
2. scaled dot product: $s_{ij} = \frac{\vec{q}_i^T\vec{k}_j}{\sqrt{m}}$, the result of the dot product is divided by $\sqrt{m}$, usually they use $d$ for $m$ here to represent the size of embedding.
3. general dot product: $\vec{q}_i^TW\vec{k}_j$, query projected in to a new space using W (but what's the difference between this $W$ and $W_q$)
4. Kernel Methods: mapping $q$ and $k$ into a new space using non-linear functions

### How to Calculate QKV
QKV are not copies of the input word vectors, instead, each of them results from a weight matrix.
$$Q = XW_q, K = XW_k, V = XW_v$$
In which $X$ is the input sentence $X = [\vec{x}_1, \vec{x}_2,..., \vec{x}_n]^T$. $\vec{x}_i$ is the same as $\vec{v_i}$ in simple self  attention formulas, and we now use $x$ instead because $v$ stands for Value now.\
Since $X$ is a $n \times m$ matrix ($m$ is the length of each word vector), $W_q, W_k, W_v$ are $m \times m$ matrices so that $Q, K, V$ remains the same shape as $X$. Meanwhile this matrix multiplication ensures that the word vectors are still independent.


## Formula
Now with the definition of QKV, we start to write down the algorithm of self attention.\
Let's start with the simplest form, and $s_j$ or $s_{ij}$ is the numerical result of $\vec{q}^T\vec{k}_1$, prior to the weight.

For one Query $q$
$$\vec{s}_1 = \vec{q}^T\vec{k}_1, \vec{s}_j = \vec{q}^T\vec{k}_j$$
For one of many Query vectors $q_1$
$$\vec{s}_{11} = \vec{q}_1^T\vec{k}_1, \vec{s}_{1j} = \vec{q}_1^T\vec{k}_j$$
Let's consider $S$ as a matrix, each row $\vec{s}_i$ is a n-dimensional vector of the weights regard to $\vec{q}_i$ times each Key $\vec{k}_j$
### Query and Key
$$
Q = 
\begin{bmatrix}
    \vec{q}_1 \\
    \vec{q}_2 \\ 
    \vdots \\ 
    \vec{q}_n
\end{bmatrix}
= 
\begin{bmatrix}
    q_{11} & q_{12} & \dots & q_{1n} \\
    q_{21} & q_{22} & \dots & q_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    q_{n1} & q_{n2} & \dots & q_{nn}
\end{bmatrix},\
K = 
\begin{bmatrix}
    \vec{k}_1 \\
    \vec{k}_2 \\ 
    \vdots \\ 
    \vec{k}_n
\end{bmatrix}
= 
\begin{bmatrix}
    k_{11} & k_{12} & \dots & k_{1n} \\
    k_{21} & k_{22} & \dots & k_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    k_{n1} & k_{n2} & \dots & k_{nn}
\end{bmatrix}
$$


### Definition of $S$
$$
S = 
\begin{bmatrix}
    \vec{s}_1 \\
    \vec{s}_2 \\
    \vdots \\
    \vec{s}_n
\end{bmatrix}
=
\begin{bmatrix}
    s_{11} & s_{12} & \dots & s_{1n} \\
    s_{21} & s_{22} & \dots & s_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    s_{n1} & s_{n2} & \dots & s_{nn} \\
\end{bmatrix}
=
\begin{bmatrix}
    \vec{q}_1^T\vec{k}_1 & \vec{q}_1^T\vec{k}_2 & \dots & \vec{q}_1^T\vec{k}_n \\
    \vec{q}_2^T\vec{k}_1 & \vec{q}_2^T\vec{k}_2 & \dots & \vec{q}_2^T\vec{k}_n \\
    \vdots & \vdots & \ddots & \vdots \\
    \vec{q}_n^T\vec{k}_1 & \vec{q}_n^T\vec{k}_2 & \dots & \vec{q}_n^T\vec{k}_n \\
\end{bmatrix}
$$

### Multiplication
In this case, we assume the similarity method is dot product.\
When we write it as matrix multiplication, the formula becomes
$$
S = QK^T = 
\begin{bmatrix}
    \vec{q}_1 \\
    \vec{q}_2 \\ 
    \vdots \\ 
    \vec{q}_n
\end{bmatrix}\
\begin{bmatrix}
    \vec{k}_1 & \vec{k}_2 & \dots & \vec{k}_n
\end{bmatrix}\
=
\begin{bmatrix}
    q_{11} & q_{12} & \dots & q_{1n} \\
    q_{21} & q_{22} & \dots & q_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    q_{n1} & q_{n2} & \dots & q_{nn}
\end{bmatrix}\
\begin{bmatrix}
    k_{11} & k_{21} & \dots & k_{n1} \\
    k_{12} & k_{22} & \dots & k_{n2} \\
    \vdots & \vdots & \ddots & \vdots \\
    k_{1n} & k_{2n} & \dots & k_{nn}
\end{bmatrix}
$$
It's not hard to realize that this matrix multiplication is the repetition of calculating $s_{11}$


### Softmax
Softmax serves as normalization processor. It is applied to __each row__ of $S$, or $s_i$ to ensure the weights related to one Query $q_i$ sum up to 1, because $w_i$ will then be weighted over $V = [\vec{v}_1, \vec{v}_2, ..., \vec{v}_n]$.
$$
W = Softmax(S) =
W = 
\begin{bmatrix}
    \vec{w}_1 \\
    \vec{w}_2 \\
    \vdots \\
    \vec{w}_n
\end{bmatrix}
=
\begin{bmatrix}
    w_{11} & w_{12} & \dots & w_{1n} \\
    w_{21} & w_{22} & \dots & w_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    w_{n1} & w_{n2} & \dots & w_{nn} \\
\end{bmatrix}
$$

### Value
$$
V = 
\begin{bmatrix}
    \vec{v}_1 \\
    \vec{v}_2 \\ 
    \vdots \\ 
    \vec{v}_n
\end{bmatrix}
= 
\begin{bmatrix}
    v_{11} & v_{12} & \dots & v_{1n} \\
    v_{21} & v_{22} & \dots & v_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    v_{n1} & v_{n2} & \dots & v_{nn}
\end{bmatrix}
$$


### Calculating Attention Output $Y$
The output $Y$ is a vector in which each column is the weighted sum between $W$ and $V$, and each column has the same number of elements as a word vector (or a Value vector).
$$
Y = 
\begin{bmatrix}
    \vec{y}_1 \\
    \vec{y}_2 \\
    \vdots \\
    \vec{y}_n
\end{bmatrix}
= 
\begin{bmatrix}
    y_{11} & y_{12} & \dots & y_{1n} \\
    y_{21} & y_{22} & \dots & y_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    y_{n1} & y_{n2} & \dots & y_{nn}
\end{bmatrix}
$$
$$
Y =
WV =
\begin{bmatrix}
    \vec{w}_1 \\
    \vec{w}_2 \\
    \vdots \\
    \vec{w}_n
\end{bmatrix}\
\begin{bmatrix}
    \vec{v}_1 \\
    \vec{v}_2 \\ 
    \vdots \\ 
    \vec{v}_n
\end{bmatrix}\
= 
\begin{bmatrix}
    w_{11} & w_{12} & \dots & w_{1n} \\
    w_{21} & w_{22} & \dots & w_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    w_{n1} & w_{n2} & \dots & w_{nn}
\end{bmatrix}\
\begin{bmatrix}
    v_{11} & v_{12} & \dots & v_{1n} \\
    v_{21} & v_{22} & \dots & v_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    v_{n1} & v_{n2} & \dots & v_{nn}
\end{bmatrix}\\
=
\begin{bmatrix}
    \sum_{i=1}^n w_{1i}v_{i1} & \sum_{i=1}^n w_{1i}v_{i2} & \dots & \sum_{i=1}^n w_{1i}v_{in} \\
    \sum_{i=1}^n w_{2i}v_{i1} & \sum_{i=1}^n w_{2i}v_{i2} & \dots & \sum_{i=1}^n w_{2i}v_{in} \\
    \vdots & \vdots & \ddots & \vdots \\
    \sum_{i=1}^n w_{ni}v_{i1} & \sum_{i=1}^n w_{ni}v_{i2} & \dots & \sum_{i=1}^n w_{ni}v_{in}
\end{bmatrix}
$$
The last equation means that the weighted mean sum of Value vectors is calculated across every element of the vectors, which corresponds to the definition of $Y^T$ that each row has the same dimension as a Value vector.\
__This also means that each dimension in the output vector is a weighted sum of information in the corresponding dimensions in all Value vectors.__

___Because $y_i$ results from the relevance of Value vectors with respect to Query $q_i$, we can say $y_i$ is a new embedding of word i with context information.___



## Attention Block and Multi-head Attention
An attention block is a structure that contains all the calculation above. The input of the block is the word vectors, and the output is the processed "word vectors".

Multi-head attention is to do several self attention calculation at the same time. These calculations are independend, and their results are concatenate and densed to produce the final attention output. From a matrix calculation perspective, multi-head attention makes QKV 3-dimensional tensors with the third dimension storing different heads' calculation.

### Concatenate and Dense
Concatenate is rather intuitive It's horizontally stacking all the outputs from attention heads.\
However, the Fead-forward layer takes input of the size of word embedding, so we need to project the concatenated output into a smaller dimension. This process is Dense.

Say we have 8 outputs each of size $2 \times 4$, therefore the concatenated output has size $2 \times 32$. The dense projection matrix will be size $32 \times 4$.
$$
Y' = YW = 
\begin{bmatrix}
    y_{11} & y_{12} & \dots & y_{1\ 32}\\
    y_{21} & y_{22} & \dots & y_{2\ 32}
\end{bmatrix}\
\begin{bmatrix}
    w_{11} & \dots & w_{14}\\
    w_{21} & \dots & w_{24}\\
    \vdots & \ddots & \vdots \\
    w_{32\ 1} & \dots & w_{32\ 4}
\end{bmatrix}\
=
\begin{bmatrix}
    y'_{11} & y'_{12} & y'_{13} & y'_{14}\\
    y'_{21} & y'_{22} & y'_{23} & y'_{24}
\end{bmatrix}\
$$

### How to Understand Multi-head Attention
Actually, for each word in a sentence, there are a lot of aspects it should focus on, each related to different words. For example, in the sentence "I enjoyed the trip last year", grammatically, the word "enjoyed" should attend to "last year", while semantically, "enjoyed" should attend to "trip". It can be hard for one attention to capture all these information, so in multi-head attention, different heads __look__ at the sentence independently and we hope them to capture different types of relevance.

# Residual Network
Residual Network was first invented in CNN for Image Recognition problems, in order to solve gradient vanishing in deep neural networks. It adds the output of several layers before directly to the current layer, so even if gradient vanishing happens and the current layer outputs 0, the output will be equal to the previous layer output, equivalent to jumping over layers in between.

Apparently, Transformer applies Residual Network (or Res Net) after every attention and feed-forward block. Res Net is mathematically proven to smooth the loss function and thus improve the training.

Since the output of attention block has the same size as the input embedding (or the residual), they can simply added together before the Layer Norm.\
This operation is reasonable, because the information of each word vector (word embedding) lies within the values of each dimension. Adding two vectors to form a new vector is reinforcing or cancelling out information in each dimension.

__By adding the residual, we are actually reducing the counter-effect other words have on this word.__

For example, the sentence "He sobs happily". Let's assume each word has two embedding dimensions, syntax and emotion.\
When the word "sobs" attend to the sentence, grammatically it should focus on "happily", because an adverb infers that "sobs" is a verb. So "happily" has 0.7 weight in computing the next embedding.\
However, we know that the emotionally, the two words are opposite, say -0.3 and 0.5.\
After computation, we surprisingly find that, at the position corresponding to "sobs", the emotion dimension is a positive 0.05!\
However, if we add the residual to the position, the emotion becomes a reasonable -2.95.

__This ensures that the embeddings are trained and updated slowly. If in several attention blocks when "sobs" attend to different words, the emotion dimensions are all positive, then the model will gradually learn that "sobs" may contain a positive meaning here.__


# Layer Normalization
Layer Normalization is a process proposed first in Neural Networks, and transformer, as we know it, takes in everything proved useful.

Layer Normalization is a bit different than it is in NN, because the output is position-wise, embeddings in different positions are separated. Therefore, the Layer Normalization is operated on each embedding independently. It's like the Softmax function in self attention, normalizing each row of output $Y$.

The formula is just z-score
$$\vec{y}_i := \frac{\vec{y}_i - \mu}{\sigma}$$

# Word Embedding in Transformer
__...__

# Positional Encoding
## Why Positional Encoding
Advantages of Self Attention is parallel computation, but comparing to RNN, it loses the intrinsic attribute of time sequence as all words are processed simultaneously. It's like a bag of words stuff in self attention. Instead of intrinsic positional feature, transformer uses positional encoding to represent the context relationship between words.

## The Development of Positional Encoding
Naturally, we think of the sequence 1, 2, ..., n to represent positions. The problem is that when the sequence gets longer, n
gets too big and can cause issues with gradients.

So we want to normalize the sequence, say divide all of them by n. Although we manage to project the sequence to (0,1], there comes a new problem. As the sequence gets longer, the difference between two positions changes. If $n=10$, the difference between the first and the second positions is 0.1, but if $n=100$, the difference is 0.01. Therefore, it's hard for machine to determine relative positions of the words when sequence length changes all the time.

Both of these methods are trying to encode the absolute position of the sequence, and the problem comes from the dependency of sequence length $n$. When we focus on relative position, we can use a periodic function to project the position sequence. For example,
$$PE(pos) = sin(\frac{pos}{\alpha})$$
where $\alpha$ controls the wavelength $L = 2\pi\alpha$.\
The $sin$ or $cos$ function has the attribute that with the period range, the positional encodings are different. But there is still limitations when we try to decide how large $\alpha$ is.\
If $\alpha$ gets too big, the difference between adjacent positional encodings gets too small. If $\alpha$ gets too small, the period is too short so that two positions not too far away can have the same positional encodings.

## The Ultimate Solution
Instead of finding the best wavelength, we want all of them (long and short ones). In transformer, positional encoding is not a number but a vector. Each vector element is the result from a sin or cos function with different wavelength. Thus the positional encoding algorithm looks like this.
$$
PE(pos, 2i) = sin(\frac{pos}{n^{2i/d}})\\
PE(pos, 2i+1) = cos(\frac{pos}{n^{2i/d}})
$$
where $n$ is the base wavelength, $d$ is the length of each positional embedding vector, $0 \leq i \leq \frac{d}{2}-1$ is the index of the vector elements. Since $sin$ and $cos$ functions are used alternately, one $i$ actually refers to two vector elements.\
From the equation we know that the wavelength ranges from $2\pi$ (when $i=0$) to less than $2n\pi$ (when $i=\frac{d}{2}-1$). In the paper of transformer, $n = 10000$.\
Since this positional encoding vector contains both long and short wavelengths, the adjacent positions are different enough while all positions are different in some ways.\
In application, $d$ is set as the length of the length of the word vector $d = m$. This allows us to add the positional encoding vectors directly to the word vectors. __Honestly I don't know why it works. Intuitively, the positional encoding should be concatenated to the word vectors as new dimensions representing positions. But history has proven their methods work so... let's keep it that way.__

## The Code to Generate Positional Encoding Image
By Jay Alammar
https://github.com/jalammar/jalammar.github.io/blob/master/notebookes/transformer/transformer_positional_encoding_graph.ipynb


```python
import numpy as np
import matplotlib.pyplot as plt
```


```python
# Code from https://www.tensorflow.org/tutorials/text/transformer
def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  
  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
  pos_encoding = angle_rads[np.newaxis, ...]
    
  return pos_encoding

```


```python

tokens = 10
dimensions = 64

pos_encoding = positional_encoding(tokens, dimensions)
print (pos_encoding.shape)

plt.figure(figsize=(12,8))
plt.pcolormesh(pos_encoding[0], cmap='viridis')
plt.xlabel('Embedding Dimensions')
plt.xlim((0, dimensions))
plt.ylim((tokens,0))
plt.ylabel('Token Position')
plt.colorbar()
plt.show()
```

# Feed-forward Network (FFN)
## How to Compute
The input of the feed-forward network is the output of attention block after adding residual and normalization. Therefore input $X$ is a matrix with the same dimension as word embedding.

What FFN in transformer is different from Forward Propagation in Neural Networks and Fully Connected Layers in CNNs is that FFN treats embedding vector at each position "separately and identically" (quoting the paper "Attention is All You Need"). So at below we focus on how one vector is processed.

Similarly, we have a weight matrix $W_1$ that multiplies the vector, and a bias $b_1$. And still we can write $b_1$ into $W_1$ using homogeneous form.
$$\vec{s} = \vec{x}_iW_1$$
Similarly, an activation function such as ReLU is applied to the computation result.
$$\vec{a} = ReLU(\vec{s})$$
Because $W_1$ doesn't necesserily maintain the dimension of $\vec{x}_i$, there is another matrix $W_2$ to reform the vector to embedding size, while adding a bias $b_2$ that we can write into $W_2$.\
So the overall formula is
$$FFN(\vec{x}) = ReLU(\vec{x}W_1)W_2 = max(0, \vec{x}W_1)W_2$$

## How to Understand
I don't know, yet.

# Encoder-Decoder Attention
In the paper introducing transformer, there are more than one multi-head attention block, instead there are 6 blocks in sequence.\
The top Encoder block produces position-wise embeddings that are of the same size as word embeddings.

These outputs are then sent to the Decoder's second attention block, the Encoder-Decoder Attention.\
In this attention, the Encoder outputs become the Key and the Value matrices, while the Query comes from the Decoder. This means that Decoder operates a weighted mean sum of information from Encoder based on each Decoder position's relevance with the Encoder positions. So each position in the output from the Encoder-Decoder Attention block is how this Decoder position attends to the Encoder information (or sentence).

# How Decoder Functions
We still have to mention the process Decoder predicts a sentence. It's good to use animation to illustrate this, but we'll just use text.

When generating a sentence, Decoder first receives a "\<START>" token. The token goes through self attention and FFN to the Encoder-Decoder Attention. These steps of computation form one Decoder block, and there can be many blocks attached in sequence.\
When the first word is predicted, it's used again as the input and helps to predict the second word. So the cycle goes on to use previously-predicted words as input to predict the next word, until the predicted token is "\<EOS>".

# Mask - Decoder Masked Self Attention
## Why Do We Need Mask
There are three cases in transformer that we need Mask.
1. The first is in Decoder self attention block. When we train the model, we want the training to be parallel.
2. The second one is when the model makes predictions.

The first is called Subsequence Mask, and the second is called Pad Mask.

### Subsequence Mask
In real prediction process, the Decoder predicts one word at a time.

For example, if the sentence we want the model to predict is "I love you", it inputs "\<START>", predicts "I", inputs "I", predicts "love", inputs "I love", predicts "I love you", inputs "I love you", predicts "\<EOS>", and the prediction ends.

When we train the model, however, we actually have all the real outputs that the model tries to predict. Instead of simulating the real prediction process, we hope to use parallel training to do all the prediction steps at once. (Parallel computation is exactly why transformer is better than RNN).

However, this comes with a problem that sentences with different lengths cannot be input in batch. Here's where Mask comes in. Instead of inputing different numbers of words, we input the sentence as a whole every time. During self attention, we mask all the words __including and after the target word being predicted__. If we manage to prevent prior words to integrate information in later words, we equivalently remove the later words from the sentence and simulate the real case. Otherwise, we say the model is "cheating" by peaking the answers before predictions.

For example, "I love you" can be masked into "x x x", "I x x", "I love x", and "I love you". These sentences can then be stuff into the Decoder at once.

The detailed computation of Mask is in the next part, but masking is just adding different mask matrices to the $QK^T$ weights. __Therefore, we change the difference in sentence length into the type of masking matrices, shifting a non-parallel problem into a parallel one.__

### Pad Mask
As we input several sentences as training examples, they very likely have different lengths. We want parallel computation, so we have to set them to the same length. We give shorter sentences numerous padding "P". However, we don't want these "P" characters to get involved in computation, so we mask the relevance computation between normal Query and "P".

Meanwhile, "P" as an input word will be transformed into a Query and calculate its relavance with the Keys, but we don't have to mask them. It's okay to let the positions with input "P" compute randomly, because when feeding into Decoder through Encoder-Decoder Attention, Query comes from Decoder, and "P" previously as a positional embedding and a Query will no longer be a problem.


## How to Implement Mask
### Subsequence Mask
Without mask, a word from the output "prefix" contains information from words generated later than itself.\
In Encoder Self Attention, we know that a Query generated from a word is compared with all Keys that are generated from all the words. So if a Query $\vec{q}_i$ operates dot product with $\vec{k}_j$ where $j > i$, position $i$ actually receives information from position $j$ that it's not suppose to. More specifically, $\vec{q}_i^T\vec{k}_j$ is a meaningful value (either positive or negative), so no matter how small it is after softmax function, $w_{ij}v_j$ still adds some information of $v_j$ to output $y_i$.

In terms of math computation, we hope that the weights of all positions where $j > i$ become 0.\
Since the weights are after-softmax, the value before softmax should be $-inf$. Thus, we mask the result of $QK^T$ into a lower-triangular matrix, while the rest are set to $-inf$. Take a $5 \times 5$ matrix for example.
$$
M = 
\begin{bmatrix}
    0 & -inf & -inf & -inf & -inf \\
    0 & 0 & -inf & -inf & -inf \\
    0 & 0 & 0 & -inf & -inf \\
    0 & 0 & 0 & 0 & -inf \\
    0 & 0 & 0 & 0 & 0
\end{bmatrix}
$$
And the Decoder Self Attention formula will be 
$$Y = Softmax(QK^T + M)V$$

### Pad Mask
The first step is Padding. Say we want a length of 5 but get an input "I love you". We should pad the last two positions with "P".\
"I love you P P"\
In self attention, we have a $5 \times 5$ relavance matrix, row index is Query, and column index is Key.\
$$
\ \ \ \ \ \ \ \ \ \ \ I\ \ \ \ \ love\ \ \ \ you\ \ \ \ \ P\ \ \ \ \ P\\
\ \ \ \ \ \ I\ false\ false\ false\ true\ ture\\
\ \ love\ false\ false\ false\ true\ ture\\
\ \ \ you\ false\ false\ false\ true\ ture\\
\ \ \ \ \ \ P\ false\ false\ false\ true\ ture\\
\ \ \ \ \ \ P\ false\ false\ false\ true\ ture\\
$$
where $true$ means need masking, and the corresponding place will be set as "-inf". From the example, we mask all the Query-Key dot product that take "P" as the Key, including those when "P" is the Query, and we leave two Query "P" drifted freely.

# The Final Linear and Softmax Layer
The Linear Layer turns the position-wise embedding, or the final output of the last Decoder block, into a long "logits vector" __with the size of the word number in this language's Word Vocabulary__ (for definition, see the next part), say there are 10,000 words in the English Word Vocabulary.\
Then the softmax turns the 10,000 elements into the probability of each English word, from "a" to "zulu", and output the probability vector.\
The prediction is made if we retain the word with the highest probability.

# Transformer Training - Loss Function
## Word Vocabulary
Word Vocabulary is generated before training, in which every word has a unique index. It's usually in dictionary form and words are listed in alphabet order from "a" to "zulu".\
From Word Vocabulary, each word can be represented with a vector of vocabulary length, where the index place equals 1, and the rest are 0. __This is the One-hot representation.__ This will be the form of our __real value__ in every training round.

## Loss Function
The output of Decoder is a vector with size of the Word Vocabulary, and each vector element is a probability of the corresponding word. Naturally, we think of the cross entropy loss function usually used along with Softmax function.
$$-\sum_iYln(y)$$


```python

```
