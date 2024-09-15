Introduction notes remark the importance of self-attention and how the paper "Attention is All You Need" changed the industry. 
Some of the key concepts which were emphasised are related to:
- How LLMs can be used for entity extraction (e.g. getting the names of people, places...). In my head, the first application it came to my mind with this feature is related to the capabilities to label documents, and then use that information for other ML learning models
- The good interaction capabilities LLMs have with APIs. This boosts the applications around all industries and for individuals that use open-source code, which can be integrated and enhanced with LLMs
- RNN has been clearly beaten by LLMs, especially as their limited compute and memory capabilities. LLMs are better in parallelising the computations, scale the computation and the attention capabilities are increased
- Attention means that each word (or I guess that token?) is related with a weight to the other parts of the text

When doing the inference of a model, the context can mark a difference. There are three categories for classifying the inference:
- Zero-shot inference: the instruction is given without any hints or example
- One-shot inference: the instruction is given with an example of how to do the tasks
- Few-shot inference: multiple possibilities of how to perform the task are given

In the inference, the model can choose how creative the answer is. The key parameters are:
* `K`: defines the number of k tokens with highest probability
* `P`: Limits the cumulative probability taken for prediction 
* `Temperature`: changes the shape of the probability distribution (the highest the temperature, the most )

##### Multi-GPU 
There are strategies to fine-tune LLMs on multiple GPUs, making use of data and models distribution:
- Distributed Data Parallel (DDP): creates a replication of the model in each GPU and distributes the data for each of the GPUs
- FSDP (zero data overlap between GPUs): it not only distributes the data, but also distributes the model across GPUs. There is a drawback, which is the need to gather and synchronise the gradients

##### Fine Tuning
Similarly to other ML models, in order to fine-tune a model you need to have a dataset (splited into training, validation and testing). The dataset usually contains a `[Prompt]` and `[Completion]`. Usually, with ~500/1000 examples should be enough to fine tune a model. However, if the model is only focused on a set of actions, other might be forgotten (catastrophic forgetting). To prevent this, some techniques can be applied, such as:
* Parameter Efficient Fine-Tuning (leaving some of the weights unchanged)
* Multitask Fine-Tuning (training the model with a mixed dataset containing multiple prompts simultaneously). Have a look at [FLAN](https://arxiv.org/abs/2210.11416)

