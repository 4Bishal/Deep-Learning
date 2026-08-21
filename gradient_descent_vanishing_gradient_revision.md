# 🧠 Gradient Descent → Vanishing Gradient Revision

> **My goal:** Understand how a neural network actually learns — not just memorize formulas.

---

## 1. 🧠 Big Picture

A neural network learns by repeatedly doing this:

```text
Current Weights + Biases
          ↓
      Forward Pass
          ↓
       Prediction
          ↓
          Loss
          ↓
    Backpropagation
          ↓
       Gradients
          ↓
   Gradient Descent
          ↓
    Update W and b
          ↓
      New W and b
          ↓
        Repeat
```

### The simple story

- **Weights and biases** control how the network transforms inputs.
- **Forward pass** uses them to produce a prediction.
- **Loss** tells us how wrong that prediction is.
- **Backpropagation** calculates how each parameter contributed to the loss.
- **Gradient descent** uses those gradients to change the parameters.
- Repeat until the model becomes better.

> **Key Idea:** Learning = repeatedly changing parameters so that the loss tends to decrease.

---

## 2. ⚙️ What Is Actually Being Updated?

The main trainable parameters are:

- **Weights** `W`
- **Biases** `b`

Together, we can call them:

$$
\theta = \{W,b\}
$$

The basic gradient-descent update is:

$$
\boxed{\theta \leftarrow \theta-\eta\nabla_{\theta}L}
$$

### What each symbol means

$$
\theta = \text{trainable parameters}
$$

$$
\eta = \text{learning rate}
$$

$$
L = \text{loss}
$$

$$
\nabla_{\theta}L
=
\text{gradient of the loss with respect to the parameters}
$$

Where:

| Symbol | Meaning |
|---|---|
| \(\theta\) | Parameters (weights + biases) |
| \(L\) | Loss |
| \(\nabla_\theta L\) | Gradient of loss with respect to parameters |
| \(\eta\) | Learning rate |

### Tiny example

Suppose:

$$
w=2,
\qquad
\eta=0.1,
\qquad
\frac{\partial L}{\partial w}=5
$$

Then:

$$
\begin{aligned}
w_{\text{new}}
&=w-\eta\frac{\partial L}{\partial w}\\
&=2-(0.1)(5)\\
&=2-0.5\\
&=\boxed{1.5}
\end{aligned}
$$

So:

```text
Old weight = 2
Gradient   = +5
Learning rate = 0.1

        ↓

New weight = 1.5
```

Because the gradient was **positive**, we moved the weight downward.

> **Remember**
>
> - Positive gradient → parameter generally decreases.
> - Negative gradient → parameter generally increases.
> - Gradient near zero → very small update.

---

## 3. 🔍 How Does a Weight Change?

Start with one simple neuron:

$$
\boxed{z=wx+b}
$$

Then an activation function may transform it:

$$
\boxed{a=f(z)}
$$

Imagine we change \(w\).

```text
Change weight
      ↓
Change z = wx + b
      ↓
Change activation a
      ↓
Change later layers
      ↓
Change prediction
      ↓
Change loss
```

### Tiny intuition

If:

$$
x=2,\qquad w=3,\qquad b=1
$$

then:

$$
z=(3)(2)+1=7
$$

If we change \(w\) from \(3\) to \(4\):

$$
z=(4)(2)+1=9
$$

So the neuron's output changes.

That changed value becomes an input to later computations, so eventually the **final prediction changes**.

> **Why does a weight matter?**  
> Because a weight controls how strongly an input contributes to the neuron's computation.

---

## 4. 🧭 What Does the Gradient Mean?

For one parameter, a derivative tells us the local sensitivity of the loss.

For example:

$$
\frac{\partial L}{\partial w}
$$

means:

> **If I slightly change \(w\), how does the loss tend to change?**

But a neural network has **many** parameters.

So we use:

$$
\boxed{\nabla_{\theta}L}
$$

This is a vector containing the partial derivatives for the parameters.

For example:

$$
\nabla_{\theta}L
=
\begin{bmatrix}
\frac{\partial L}{\partial w_1}\\
\frac{\partial L}{\partial w_2}\\
\frac{\partial L}{\partial b_1}\\
\vdots
\end{bmatrix}
$$

```text
Gradient
   ↓
[ ∂L/∂w₁
  ∂L/∂w₂
  ∂L/∂w₃
  ∂L/∂b₁
  ... ]
```

> **Key Idea:** Each component tells us how sensitive the loss is to that particular parameter.

The gradient points toward the direction of **steepest local increase** in loss.

Therefore gradient descent moves in the opposite direction.

---

## 5. 🔄 Iterative Weight Updates

One update is not enough.

Training looks like:

```text
Start with parameters
        ↓
Calculate loss
        ↓
Calculate gradients
        ↓
Update parameters
        ↓
Calculate loss again
        ↓
Calculate gradients again
        ↓
Update again
        ↓
...
```

Mathematically:

$$
\boxed{
\theta_{t+1}
=
\theta_t-\eta\nabla_{\theta}L
}
$$

The subscript \(t\) simply means:

> **parameters at the current training step.**

---

## 6. 🧩 One Complete Training Step

Think of one training step as:

```text
Input
  ↓
Forward Pass
  ↓
Prediction
  ↓
Loss
  ↓
Backpropagation
  ↓
Gradients
  ↓
Gradient Descent
  ↓
Update W, b
```

### What changes?

Only the **parameters** are updated:

```text
W_old, b_old
     ↓
 gradient descent
     ↓
W_new, b_new
```

The next forward pass uses these new values.

---

## 7. 🔗 Backpropagation vs Gradient Descent

These two are related, but they are **not the same thing**.

### Backpropagation

> **Calculates the gradients.**

It answers:

> "How much does the loss change with respect to each parameter?"

### Gradient Descent

> **Uses those gradients to update the parameters.**

It answers:

> "Given these gradients, how should I change the parameters?"

```text
Backpropagation
      ↓
Find gradients
      ↓
Gradient Descent
      ↓
Update parameters
```

> **Remember:**  
> **Backpropagation = gradient calculation**  
> **Gradient Descent = parameter update**

---

## 8. 📦 Batch Gradient Descent

Batch Gradient Descent uses the **entire dataset** to calculate the gradient before making an update.

If the dataset contains \(N\) samples:

```text
N samples
   ↓
Calculate gradient using all N
   ↓
1 parameter update
```

Therefore:

$$
\boxed{1\text{ update per epoch}}
$$

### Why?

One epoch = one complete pass through the dataset.

Batch GD waits until that complete dataset has contributed to the gradient.

---

## 9. 🎲 Stochastic Gradient Descent (SGD)

SGD uses **one sample** to calculate a gradient and update the parameters.

```text
Sample 1 → gradient → update
Sample 2 → gradient → update
Sample 3 → gradient → update
...
```

For \(N\) samples:

$$
\boxed{N\text{ updates per epoch}}
$$

### Why is SGD noisy?

The full dataset gives us a fuller picture of the loss:

$$
\nabla L
$$

But SGD uses only one sample:

$$
\nabla L_i
$$

Generally:

$$
\boxed{\nabla L_i\neq\nabla L}
$$

So each update is a **noisy estimate** of the full-data gradient.

```text
Full gradient
     ↓
More complete direction

Single-sample gradient
     ↓
Approximate direction
     ↓
Noise / fluctuation
```

This noise can sometimes help optimization move through **flat regions, saddle regions, or shallow local minima** instead of following one perfectly smooth path.

> **Important:** This does **not** mean SGD is guaranteed to escape every local minimum.

---

## 10. 🧺 Mini-Batch Gradient Descent

Mini-batch GD sits between Batch GD and SGD.

Instead of:

- all samples → Batch GD
- one sample → SGD

we use a small group of samples.

If:

$$
B=\text{batch size}
$$

then approximately:

$$
\boxed{
\text{updates per epoch}
\approx
\frac{N}{B}
}
$$

### Example

```text
Dataset = 1000 samples
Batch size = 10

1000 / 10 = 100 updates per epoch
```

The important distinction:

> **Batch size = number of samples used for ONE update.**

NOT:

> "Batch size = number of updates per epoch."

---

## 11. 📊 Epoch vs Update

These terms are easy to mix up.

### Batch size

> How many samples are used to calculate **one update**?

### Update

> One change to the model parameters.

### Epoch

> One complete pass through the training dataset.

### Example

```text
Dataset = 1000
Batch size = 10

One update:
10 samples → gradient → update

One epoch:
1000 samples processed
      ↓
100 updates
```

So:

$$
\boxed{
\frac{1000}{10}=100
\text{ updates per epoch}
}
$$

### Quick comparison

| Method | Samples per update | Approx. updates per epoch |
|---|---:|---:|
| Batch GD | \(N\) | 1 |
| SGD | 1 | \(N\) |
| Mini-batch | \(B\) | \(N/B\) |

---

## 12. 🎚️ Learning Rate

The learning rate controls **how large the parameter update is**.

$$
\boxed{
\theta_{\text{new}}
=
\theta-\eta\nabla_{\theta}L
}
$$

### Too small

```text
Tiny updates
    ↓
Slow learning
    ↓
Many iterations / epochs
```

### Too large

```text
Large updates
    ↓
Overshooting
    ↓
Oscillation or divergence
```

### Reasonable learning rate

```text
Useful step size
    ↓
Steady progress
```

> **Most important distinction:**
>
> **Gradient = direction / sensitivity**  
> **Learning rate = update size**

---

## 13. 🕳️ Vanishing Gradient

This happens when gradients become **extremely small**, especially while flowing backward through many layers.

The key reason is the **chain rule**.

For a simple chain:

$$
w\rightarrow z\rightarrow a\rightarrow L
$$

the derivative can be written as:

$$
\boxed{
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial a}
\frac{\partial a}{\partial z}
\frac{\partial z}{\partial w}
}
$$

Across many layers, this becomes a much longer product.

If many terms are smaller than 1:

$$
0.2\times0.2\times0.2\times0.2\times\cdots
$$

the result can become tiny very quickly.

### The learning consequence

```text
Small derivatives
      ↓
Repeated multiplication
      ↓
Tiny gradient
      ↓
Tiny parameter update
      ↓
Early layers barely change
      ↓
Very slow learning
```

> **Key Idea:** Vanishing gradients are mainly a **gradient-flow problem**.

---

## 14. ⛓️ Chain Rule Explanation

The chain rule lets us determine how a change at one point affects something much farther away.

For example, if:

$$
w\rightarrow z\rightarrow a\rightarrow L
$$

then:

$$
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial a}
\cdot
\frac{\partial a}{\partial z}
\cdot
\frac{\partial z}{\partial w}
$$

Imagine:

```text
w → z → activation → later layer → prediction → loss
```

The effect of \(w\) on the loss passes through all those steps.

So conceptually:

$$
\text{Effect on loss}
=
\text{effect at step 1}
\times
\text{effect at step 2}
\times
\cdots
$$

If many effects are small:

$$
\text{small}\times\text{small}\times\text{small}
\rightarrow\text{very small}
$$

That is the heart of the vanishing-gradient problem.

---

## 15. 🟣 Sigmoid + Vanishing Gradient

The sigmoid function is:

$$
\boxed{
\sigma(z)=\frac{1}{1+e^{-z}}
}
$$

Its output satisfies:

$$
\boxed{
0<\sigma(z)<1
}
$$

Its derivative is:

$$
\boxed{
\sigma'(z)
=
\sigma(z)\bigl(1-\sigma(z)\bigr)
}
$$

The maximum derivative occurs at \(\sigma(z)=0.5\):

$$
\sigma'(z)
=
0.5(1-0.5)
=
\boxed{0.25}
$$

Therefore:

$$
\boxed{
0<\sigma'(z)\leq0.25
}
$$

The derivative approaches \(0\) when the sigmoid saturates near \(0\) or \(1\).

### ⚠️ Important correction

Do **not** confuse these:

```text
Sigmoid output:
0 < σ(z) < 1

Maximum sigmoid derivative:
σ'(z) = 0.25
```

The sigmoid output can get close to `1`, but its derivative can never reach `0.5`.

### Why does sigmoid contribute to vanishing gradients?

During backpropagation, sigmoid derivatives are multiplied.

A simplified picture:

```text
0.25 × 0.25 × 0.25 × ...
          ↓
       becomes tiny
```

And sigmoid has another problem: **saturation**.

When its output is close to \(0\) or \(1\):

$$
\sigma'(z)\approx0
$$

So the gradient becomes even smaller.

```text
Sigmoid saturates
      ↓
Derivative ≈ 0
      ↓
Small gradient
      ↓
Poor gradient flow
```

---

## 16. 🧱 Why Early Layers Suffer

Think about the backward path:

```text
Loss
 ↓
Output Layer
 ↓
Hidden Layer
 ↓
Hidden Layer
 ↓
Hidden Layer
 ↓
Early Hidden Layer
 ↓
Input
```

The gradient for an early layer has to pass through **more derivative factors**.

Therefore:

```text
Later layer
→ fewer factors

Earlier layer
→ more factors
→ more chances for gradients to shrink
```

> **Key Idea:** Early layers are generally more vulnerable to vanishing gradients.

---

## 17. ❌ Does the Whole Network Stop Learning?

**Not necessarily.**

Vanishing gradients do not mean:

> "Every parameter becomes exactly zero and the whole network stops."

A more accurate picture is:

```text
Later layers
→ may still receive useful gradients
→ can continue learning

Earlier layers
→ receive extremely small gradients
→ learn very slowly
```

So the problem is often:

> **Learning becomes extremely slow in earlier layers.**

---

## 18. 🟩 ReLU

ReLU means **Rectified Linear Unit**.

$$
\boxed{ReLU(z)=\max(0,z)}
$$

Simple behavior:

```text
z < 0  → 0
z > 0  → z
```

Its derivative is:

$$
\boxed{
ReLU'(z)=
\begin{cases}
0, & z<0\\[4pt]
1, & z>0
\end{cases}
}
$$

At \(z=0\), the derivative is not uniquely defined; implementations choose a convention for this single point.

---

## 19. 💡 Why ReLU Helps

For positive inputs:

$$
\boxed{ReLU'(z)=1}
$$

So ReLU does not shrink the gradient in its positive region.

Compare:

```text
Sigmoid:
0.25 × 0.25 × 0.25 × ...
→ tiny

ReLU (positive region):
1 × 1 × 1 × ...
→ no shrinking from ReLU derivatives
```

This is one major reason ReLU became so useful in deep neural networks.

> **Important nuance:**  
> ReLU **reduces** the vanishing-gradient problem. It does not eliminate every possible source of vanishing gradients in a deep network.

---

## 20. ☠️ Dying ReLU

ReLU also has a weakness.

For:

$$
z<0
$$

we get:

$$
ReLU(z)=0
$$

and:

$$
ReLU'(z)=0
$$

So the gradient through that neuron can become zero.

```text
Negative input
      ↓
ReLU output = 0
      ↓
Derivative = 0
      ↓
No useful gradient through that path
      ↓
Neuron may stop learning
```

A neuron that remains in this inactive region for many inputs can become a **dead / dying ReLU**.

### The trade-off

```text
Sigmoid
   ↓
Smooth but can strongly shrink gradients

ReLU
   ↓
Better gradient flow in positive region
   ↓
But can have dying neurons in negative region
```

---

## 21. ⚖️ Sigmoid vs ReLU

| Property | Sigmoid | ReLU |
|---|---|---|
| Output | \(0<\sigma(z)<1\) | \(\max(0,z)\) |
| Positive-region derivative | Can be small | \(1\) |
| Saturation | Yes | No for \(z>0\) |
| Vanishing-gradient risk | Higher | Lower in positive region |
| Negative input | Small positive output | Exactly 0 |
| Main issue | Saturation / small derivatives | Dying ReLU |

> **Simple mental model:**
>
> **Sigmoid:** gradient can become too small.  
> **ReLU:** gradient flows well when active, but can become zero when inactive.

---

## 22. 🧠 How Everything Connects

This is the main mental model to keep.

```text
              TRAINING
                 │
                 ▼
        Current W + b
                 │
                 ▼
          ┌─────────────┐
          │ Forward Pass│
          └─────────────┘
                 │
                 ▼
             Prediction
                 │
                 ▼
               Loss
                 │
                 ▼
          Backpropagation
                 │
                 ▼
             Gradients
                 │
                 ▼
        Learning Rate η
                 │
                 ▼
       Gradient Descent
                 │
                 ▼
            Update W + b
                 │
                 ▼
           New W + b
                 │
                 └──────────► Repeat
```

### Where activation functions fit

Activation functions affect **both directions**:

```text
Forward Pass
     ↓
Activation function
     ↓
Network output
```

and during backpropagation:

```text
Activation function
     ↓
Its derivative
     ↓
Gradient flow
     ↓
Parameter updates
     ↓
Learning
```

So an activation function is **not just a forward-pass decision**.

Its derivative directly affects how well gradients can travel backward.

---

### 🔗 The Full Connection

```text
Weights + Biases
       ↓
Forward Pass
       ↓
Prediction
       ↓
Loss
       ↓
Backpropagation
       ↓
Gradient
       ↓
Learning Rate
       ↓
Parameter Update
       ↓
New Weights + Biases
       ↓
Repeat
```

And inside the gradient calculation:

```text
Activation Function
       ↓
Derivative
       ↓
Gradient Flow
       ↓
Parameter Update
       ↓
Learning
```

This connects the topics:

```text
Gradient Descent
      ↕
Gradients
      ↕
Backpropagation
      ↕
Chain Rule
      ↕
Activation Derivatives
      ↕
Vanishing Gradient
      ↕
Sigmoid / ReLU
      ↕
How Well the Network Learns
```

> **Big Picture:**  
> The network learns because the loss produces a signal that is propagated backward into gradients, and those gradients tell gradient descent how to change the parameters.

---

## 23. 🧮 Mathematical Notation Cheat Sheet

These are the formulas I should recognize immediately when revising.

### Neuron

$$
\boxed{z=wx+b}
$$

For many inputs:

$$
\boxed{
z=\sum_{i=1}^{n}w_i x_i+b
}
$$

### Activation

$$
\boxed{a=f(z)}
$$

### Gradient

$$
\boxed{
\nabla_{\theta}L
=
\begin{bmatrix}
\frac{\partial L}{\partial \theta_1}\\
\frac{\partial L}{\partial \theta_2}\\
\vdots\\
\frac{\partial L}{\partial \theta_n}
\end{bmatrix}
}
$$

### Gradient-descent update

$$
\boxed{
\theta_{\text{new}}
=
\theta_{\text{old}}
-
\eta\nabla_{\theta}L
}
$$

### One parameter

$$
\boxed{
w_{\text{new}}
=
w_{\text{old}}
-
\eta\frac{\partial L}{\partial w}
}
$$

### Chain rule

For:

$$
w\rightarrow z\rightarrow a\rightarrow L
$$

we get:

$$
\boxed{
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial a}
\frac{\partial a}{\partial z}
\frac{\partial z}{\partial w}
}
$$

### Sigmoid

$$
\boxed{
\sigma(z)=\frac{1}{1+e^{-z}}
}
$$

$$
\boxed{
\sigma'(z)=\sigma(z)(1-\sigma(z))
}
$$

$$
\boxed{
0<\sigma(z)<1
}
$$

$$
\boxed{
0<\sigma'(z)\leq0.25
}
$$

### ReLU

$$
\boxed{
ReLU(z)=\max(0,z)
}
$$

$$
\boxed{
ReLU'(z)=
\begin{cases}
0,&z<0\\
1,&z>0
\end{cases}
}
$$

### Mini-batch updates

If:

$$
N=\text{number of training samples}
$$

and:

$$
B=\text{batch size}
$$

then approximately:

$$
\boxed{
\text{updates per epoch}
=
\frac{N}{B}
}
$$

when \(B\) divides \(N\) exactly.

> **Notation reminder:**  
> \(w\) = one weight, \(W\) = collection/matrix of weights, \(b\) = bias, \(L\) = loss, \(\eta\) = learning rate, \(\theta\) = all trainable parameters.

---

## 24. ⚡ One-Minute Revision

### Gradient Descent

$$
\boxed{\theta\leftarrow\theta-\eta\nabla_\theta L}
$$

Moves parameters in the direction that locally reduces loss.

### Gradient

> Tells how sensitive the loss is to each parameter.

### Backpropagation

> Calculates those gradients.

### Gradient Descent

> Uses those gradients to update parameters.

### Learning Rate

> Controls update size.

### Batch GD

> All samples → one gradient → one update.

### SGD

> One sample → one gradient estimate → one update.

### Mini-Batch GD

> Small batch → one gradient estimate → one update.

### Epoch

> One complete pass through the dataset.

### Vanishing Gradient

> Repeated multiplication of small derivatives can make gradients extremely small.

### Sigmoid

$$
0<\sigma(z)<1
$$

$$
0<\sigma'(z)\leq0.25
$$

Saturation can make the derivative close to zero.

### ReLU

$$
\boxed{
ReLU(z)=\max(0,z)
}
$$

Positive region:

ReLU'(z)=1
$$

$$
Better gradient flow, but possible dying neurons.

### Dying ReLU

> ReLU neuron stays inactive with zero derivative and may stop learning.

---

## 25. 🚀 Next Topics

These are the natural next pieces of the optimization puzzle:

### Weight Initialization
→ How should weights be initialized before training?

### Xavier / Glorot Initialization
→ Helps maintain a reasonable activation and gradient scale.

### He Initialization
→ Designed especially for ReLU-style networks.

### Batch Normalization
→ Helps stabilize activations during training.

### Momentum
→ Helps optimization move more smoothly and can reduce oscillation.

### Adam
→ An adaptive optimizer that combines momentum-like behavior with adaptive learning rates.

---

## 🎯 Final Mental Model

If I remember only one thing:

```text
The network makes a prediction
          ↓
Loss tells how wrong it is
          ↓
Backpropagation asks:
"Which parameters caused this loss,
and by how much?"
          ↓
Gradients answer that question
          ↓
Gradient Descent changes the parameters
          ↓
Forward pass happens again
          ↓
Hopefully loss decreases
```

And:

```text
Activation function
        ↓
Its derivative
        ↓
How gradients flow backward
        ↓
How parameters get updated
        ↓
How the network learns
```

> **🧠 Neural-network learning is basically a repeated feedback loop:**
>
> **predict → measure error → trace responsibility backward → update → repeat.**
