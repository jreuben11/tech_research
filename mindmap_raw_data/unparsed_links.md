Fair and green hyperparameter optimization via multi-objective and multiple information source Bayesian optimization https://link.springer.com/article/10.1007/s10994-024-06515-0 It has been recently remarked that focusing only on accuracy in searching for optimal Machine Learning models amplifies biases contained in the data, leading to unfair predictions and decision supports. Recently, multi-objective hyperparameter optimization has been proposed to search for Machine Learning models which offer equally Pareto-efficient trade-offs between accuracy and fairness. Although these approaches proved to be more versatile than fairness-aware Machine Learning algorithms—which instead optimize accuracy constrained to some threshold on fairness—their carbon footprint could be dramatic, due to the large amount of energy required in the case of large datasets. We propose an approach named FanG-HPO: fair and green hyperparameter optimization (HPO), based on both multi-objective and multiple information source Bayesian optimization. FanG-HPO uses subsets of the large dataset to obtain cheap approximations (aka information sources) of both accuracy and fairness, and multi-objective Bayesian optimization to efficiently identify Pareto-efficient (accurate and fair) Machine Learning models. Experiments consider four benchmark (fairness) datasets and four Machine Learning algorithms, and provide an assessment of FanG-HPO against both fairness-aware Machine Learning approaches and two state-of-the-art Bayesian optimization tools addressing multi-objective and energy-aware optimization.
An encoding approach for stable change point detection https://link.springer.com/article/10.1007/s10994-023-06510-x Without imposing prior distributional knowledge underlying multivariate time series of interest, we propose a nonparametric change-point detection approach to estimate the number of change points and their locations along the temporal axis. We develop a structural subsampling procedure such that the observations are encoded into multiple sequences of Bernoulli variables. A maximum likelihood approach in conjunction with a newly developed searching algorithm is implemented to detect change points on each Bernoulli process separately. Then, aggregation statistics are proposed to collectively synthesize change-point results from all individual univariate time series into consistent and stable location estimations. We also study a weighting strategy to measure the degree of relevance for different subsampled groups. Simulation studies are conducted and shown that the proposed change-point methodology for multivariate time series has favorable performance comparing with currently available state-of-the-art nonparametric methods under various settings with different degrees of complexity. Real data analyses are finally performed on categorical, ordinal, and continuous time series taken from fields of genetics, climate, and finance.


V-JEPA: The next step toward Yann LeCun’s vision of advanced machine intelligence (AMI) https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/ video https://www.youtube.com/watch?v=7UkJPwz_N_0&ab_channel=YannicKilcher






Deploy an AI Coding Assistant with NVIDIA TensorRT-LLM and NVIDIA Triton https://developer.nvidia.com/blog/deploy-an-ai-coding-assistant-with-nvidia-tensorrt-llm-and-nvidia-triton/ 
- identify common base architecture - eg StarCoder uses GPT - use scripts from example folder: https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/gpt
- `hf_gpt_convert.py`: convert model weights format from Hugging Face Transformer to FasterTransformer
- `build.py` compile the model into a TensorRT engine, `--model_dir` using the converted weights
- `run.py` run the model
- 




two types of TensorRT runtimes: a standalone runtime which has C++ and Python bindings, and a native integration into TensorFlow. ONNXClassifierWrapper simplified wrapper which calls the standalone runtime

https://github.com/NVIDIA/JAX-Toolbox

Statistical dispersion https://en.wikipedia.org/wiki/Statistical_dispersion
refresher: Cross-Entropy Loss https://towardsdatascience.com/cross-entropy-loss-function-f38c4ec8643e
https://en.wikipedia.org/wiki/Loss_function
https://en.wikipedia.org/wiki/Activation_function
https://en.wikipedia.org/wiki/Softmax_function
https://en.wikipedia.org/wiki/Exponential_family
Authortiarian Leftists: From Stalin’s Moscow Show Trials to Trump’s impeachment & the relentless lawfare to remove Bibi from office



Roko's Basilisk: "If agent A has the source code to agent B, then A could analyze B's source code to determine whether B has property X" https://www.lesswrong.com/posts/B7XovxH9PunrBr4NB/rice-s-theorem-says-that-ais-can-t-determine-much-from
Rice's Theorem says that AIs can't determine much from studying AI source code https://www.lesswrong.com/posts/B7XovxH9PunrBr4NB/rice-s-theorem-says-that-ais-can-t-determine-much-from


Monte Carlo tree search (MCTS) https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
searching for the best move in a game. 4 steps:
1. Starting at root node of the tree, select optimal child nodes until a leaf node is reached.
2. Expand the leaf node and choose one of its children.
3. Play a simulated game starting with that node.
4. Use the results of that simulated game to update the node and its ancestors.






PAC Generalization via Invariant Representations https://openreview.net/attachment?id=zAgouWgI7b&name=pdf transformations of covariates such that the best model on top of the representation is invariant across training environments. In the context of linear Structural Equation Models (SEMs), learn models with out-of-distribution guarantees, i.e., robust to interventions. 

Speeding Up Bellman Ford via Minimum Violation Permutations https://openreview.net/attachment?id=mrykt39VUw&name=pdf for computing single source shortest paths in graphs with negative edge-weights. Minimum Violation Permutations: approximation guarantees for leveraging auxiliary information from similar instances to improve running time

Statistical Indistinguishability of Learning Algorithms https://openreview.net/attachment?id=LxodbQa62n&name=pdf  study similarity of outcomes of learning rules through the lens of the Total Variation (TV) distance of distributions. information-theoretic equivalences with existing algorithmic stability notions such as replicability and approximate differential privacy. 

Bandit Online Linear Optimization (OLO) with Hints and Queries https://openreview.net/attachment?id=SgeIqUvo4w&name=pdf provide algorithm  access to external information about unknown cost vector: cannot improve the standard regret bounds of O˜(√T) by using hints, but queries with correct responses can achieve O(log T) regret. make more robust via feedback on quality of responses.

Ewald-based Long-Range Message Passing for Molecular Graphs https://openreview.net/attachment?id=vd5JYAml0A&name=pdf
- MPNNs learn potential energy surfaces from molecular data: favorable scaling with system size partly relies upon a spatial distance limit on messages: locality is a useful inductive bias, but impedes learning of long-range interactions such as electrostatics and van der Waals forces. 
- Ewald message passing: a nonlocal Fourier space scheme which limits interactions via a cutoff on frequency instead of distance: augmentation
on top of existing MPNN, computationally inexpensive and agnostic to architectural details

Fast (1 + ε)-Approximation Algorithms for Binary Matrix Factorization https://openreview.net/attachment?id=Iey50XHA3g&name=pdf approximate A as a product of lowrank factors U ∈ {0, 1}^ n×k and V ∈ {0, 1}^k×d -> find U and V that minimize the Frobenius loss ||UV − A||^2_F. generalize to other common variants of the BMF problem

Refined Regret for Adversarial MDPs with Linear Function Approximation https://openreview.net/attachment?id=7WdMBofQFx&name=pdf
- where the loss functions can change arbitrarily over K episodes and the state space can be arbitrarily large. assume that the Q-function of any policy is linear in some known features, that is, a linear function approximation exists.
- improve the regret upper bound to Oe(√K): 
- refined analysis of Follow-theRegularized-Leader (FTRL) algorithm with the log-barrier regularizer allows loss estimators to be arbitrarily negative 
- magnitude-reduced loss estimator removes polynomial dependency - better alternative to Matrix Geometric Resampling procedure

Knowledge Graph Embeddings
comparison of activation functions, loss functions, optimizers
linkedin likes

reward modelling/learning
- learning reward functions from external feedback signals, rather than trying to specify them by hand
- graph reward function to visualize it is continuous 
- tune weights of existing rewards or add more reward terms -> complexity can hurt perf
-  hyperparameter tuning - heuristic
- simpler rewards -> easier tuning and better performance. reward shaping
- increase environment transitions collected for each update while tuning other hyperparameters
- curriculum with domain randomization: add more noise as performance improves
- trial-and-error reward design - unsolved problem
- Scalable agent alignment via reward modeling (2018) https://arxiv.org/abs/1811.07871 meta-RL learn a reward function from interaction with humans, optimizing the learned reward function with RL -> alignment problems !
- Reward (Mis)design for autonomous driving https://www.sciencedirect.com/science/article/pii/S0004370222001692#se0040 8 Sanity checks for reward functions
  1.	Unsafe reward shaping:	If reward includes guidance on behavior that deviates from only measuring desired outcomes, reward shaping exists.	Separately define the true reward function and any shaping reward. Report both true return and shaped return. Change it to an applicable safe reward shaping method. Remove reward shaping.
  2.	Mismatch in people's and reward function's preference orderings:	If there is human consensus that one trajectory is better than another, the reward function should agree.	Change the reward function to align its preferences with human consensus.
  3.	Undesired risk tolerance via indifference points:	Assess a reward function's risk tolerance via indifference points and compare to a human-derived acceptable risk tolerance.	Change reward function to align its risk tolerance with human-derived level.
  4.	Learnable loophole(s):	If learned policies show a pattern of undesirable behavior, consider whether it is explicitly encouraged by reward.	Remove encouragement of the loophole(s) from the reward function.
  5.	Missing attribute(s):	If desired outcomes are not part of reward function, it is indifferent to them.	Add missing attribute(s).
  6.	Redundant attribute(s):	Two or more reward function attributes include measurements of the same outcome.	Eliminate redundancy.
  7.	Trial-and-error reward design:	Tuning the reward function to improve RL agents' performances has unexamined consequences.	Only use observations of behavior to improve the reward function's measurement of task outcomes or to tune separately defined shaping reward.
  8.	Incomplete description of problem specification:	Missing descriptions of reward function, termination conditions, discount factor, or time step duration may indicate insufficient consideration of the problem specification.	In research publications, write the full problem specification and why it was chosen. The process might reveal issues.
Perils of Trial-and-Error Reward Design: Misdesign through Overfitting and Invalid Task Specifications https://ojs.aaai.org/index.php/AAAI/article/view/25733 reward functions that align exactly with a task's true performance metric are often necessarily sparse (eg 1 upon success & 0 otherwise) -> hard to learn from. replace with alternative dense reward functions. weighing relative goodness of individual state-action pairs leads to misdesign through invalid task specifications, since RL algorithms use cumulative reward github.com/serenabooth/reward-design-perils
Designing Rewards for Fast Learning https://arxiv.org/abs/2205.15400 constraint optimization with linear-programming: choose state-based rewards that maximize action gap, minimizing horizon subjective discount to encourage agents to make optimal decisions with less lookahead. When rewarding subgoals along the target trajectory, rewards should gradually increase as the goal gets closer
- Causes of RL agent reaching a more optimal policy but not continuing to improve https://www.reddit.com/r/reinforcementlearning/comments/14w6qsl/causes_of_rl_agent_reaching_a_more_optimal_policy/ - learning rate too large -> policy doesn't stablize around the optimal one. entropy too large -> epsilon decays too quickly. verify by plotting epsilon vs. episode correlation. adjust decay rate according to total number of episodes. dense reward function -> agent getting stuck in a local optima without enough incentive to explore past that local optima state


Challenges that adjacency lists + GNNs solve https://www.v7labs.com/blog/graph-neural-networks-guide#challenges-in-analyzing-a-graph
1. Traditional algorithmic methods 
  - require prior knowledge of the graph 
  - no graph level classification; just search, shortest path, spanning tree, clustering
2. problems with adjacency matrix for graph representation
  - very sparse matrices are inefficient
  - not permutation invariant: can be multiple adjacency matrices representing the same graph
3. dynamic graphs: kernel sizes and dilation rates are graph specific, so the following will not generalize:
  - depthwise separable convolutions to handle the dynamic data dimensions
  - dilated convolutions to increase the receptive field
GNN architecture: 
- Graph In, Graph Out
- spectral: orthogonally projected from spatial domain to spectral domain using discrete Graph Fourier transform (a dot product of eigenvalues with a function f that maps the graph vertices to some real number). feature matrix U of eigenvalues obtained from a spectral decomposition of a Laplacian matrix. graph convolution in the spectral domain is simply a multiplication of the spectral input signal and the spectral convolution kernel
- Spatial: computationally less expensive, support dynamic graphs directed or undirected. Message Passing Neural Networks (MPNN) and Graph Attention Networks (GAT)
- Sampling methods take into account the challenge of scalability for both. sample a subset of nodes: GraphSage uniform node sampling, DeepWalk random node sequences

The Weisfeiler-Lehman Isomorphism Test https://davidbieber.com/post/2019-05-10-weisfeiler-lehman-isomorphism-test/
Breaking the Limits of Message Passing GNNs https://arxiv.org/abs/2106.04319 
- linear complexity with respect to number of nodes in sparse graphs limited to 1st order Weisfeiler-Lehman test (1-WL)
- spectral-GNN using non-linear custom function of eigenvalues and masked with an arbitrary large receptive field -> experimentally as powerful as a 3-WL model, while remaining spatially localized
- custom filter functions -> outputs can have various frequency components that allow the convolution process to learn different relationships between a given input graph signal and its associated properties
- 3-WL equivalent GNNs: O(n^3) time complexity, O(n^2) space complexity  

graph embeddings - random samplings + message passing -> aggregations

Google GQA (Grouped Query Attention): Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints https://arxiv.org/abs/2305.13245 Multi-query attention (MQA), which only uses a single key-value head, drastically speeds up decoder inference. However, MQA can lead to quality degradation, and moreover it may not be desirable to train a separate model just for faster inference. We (1) propose a recipe for uptraining existing multi-head language model checkpoints into models with MQA using 5% of original pre-training compute, and (2) introduce grouped-query attention (GQA), a generalization of multi-query attention which uses an intermediate (more than one, less than number of query heads) number of key-value heads. We show that uptrained GQA achieves quality close to multi-head attention with comparable speed to MQA.



Gut–brain axis https://en.wikipedia.org/wiki/Gut%E2%80%93brain_axis

   
W3C Semantic Web working draft updates https://www.w3.org/blog/news/archives/9939





graphql over grpc ?

 
TO SCRAPE: 
1. FastAPI + Starlette + Uvicorn + Pydantic
2. Arrow + DataFusion
3. HF Transformer Agents
4. CUDA Examples
CuDF vs Polars 

A concept mapping for embeddings ?
https://makersuite.google.com/app/apikey
chromaDB vs Milvus










HuggingFace
- Diffusion["tensorflow"] ?
Diffusion: UNet -> VAE ?, Transformer vs UNet ?



# How I got into AI
- as a teen in the 80s, I read a lot of Science-fiction !
- after studying Computer Science in the early 90s, I landed a job at Fujitsu and worked on CASE tools and a Lisp machine implementation amongst other things
- In the mid 90s, I read "Practical Neural Network Recipies in C++" https://www.amazon.com/Practical-Neural-Network-Recipies-C/dp/0124790402/ref=tmm_pap_swatch_0?_encoding=UTF8&qid=&sr= (I still own this) - no GPUs then. A lot of for loops !
- in the late 90s I worked with OLAP cubes and Data Mining: Clustering and Decision Trees on SQL Server - my first real exposure to ML
- 2000-2004: worked on semantic web tech, studied MSc Bioinformatics - many sequence alignment algorithms + SVMs
- 2005-2010: AI, a Modern Aproach, foundational maths (linear algebra, advanced calculus, graph theory, probability and stats, stochastic calculus [quant fi]) and philosophy. Also learned the .NET EnCog neural net framework at the time
- 2011-2012: Numerical Methods, HPC, .NET resistence from peers "this is not LOB or WebDev", Hadoop Big Data Map Reduce. 
- 2013-2015: break - C++ 3D game dev, learned DirectX, some CUDA, some Python. read Causality
- 2016-2018: AdTech - Spark MLLib, Python Numpy & SKLearn, BigQuery + Apache Beam, Neo4j + SparQL on Wikidata + GraphQL + NetworkX. learned Google Data Science course, deep dive of Tensorflow API + packt books, ML Mastery blog, 3Blue1Brown multivariate calculus
- 2018-2020: Kami: 3 years working on a ChatGPT competitor: reading DL research papers + managing distributed DL teams, Transformers, RL, GNNs. read the DL book, Max Lapan's RL book, Graph Representation Learning book, problog
- 2021: Managed MLOps team on KubeFlow, Milvus, Dask, Trino
- 2022: huggingface course, Pinecone courses on NLP & LangChain
- 2023: learned JAX, deep dive RL / GNNs, revised Mathematical Monk ML videos, started to deep dive CUDA
- 2016-2018: adtech proto-RAG - Neo4j KG populated via NER + WikiData SparQL
- 2019-2021:  ChatGPT competitior - encorporating STT + emotions from melspec + MTurk proto-RLHF Rasa dialog flow + voice style transfer
- 2021-2024: MLOps + DataLakehouse pipeline



https://3b1b.github.io/manim/

Drinking from the sea of knowledge - tradeoff: read and grok 10 things vs code & debug 1 thing

graph tokenizers / video tokenizers / event stream tokenizers ???
metalearning to interpret learned embeddings - video stream embeddings from gopro or dashcam dataset. then find extreme embeddings (few outliers) and do scene analysis

Semi-Supervised Graph-to-Graph Translation arxiv.org/abs/2103.08827

Kalidokit https://github.com/yeemachine/kalidokit Blendshape and kinematics calculator for Mediapipe/Tensorflow.js Face, Eyes, Pose, and Finger tracking models.