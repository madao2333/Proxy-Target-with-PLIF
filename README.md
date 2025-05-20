# Targeting-Spiking-Actor-Networks-in-Reinforcement-Learning
PyTorch implementation of NeurIPS 2025 submission 3118 "Proxy Target: Bridging the Gap Between Discrete Spiking Neural Networks and Continuous Control".


Experiments can be run by calling:
```
python main.py --env Ant-v4  --proxy True --spiking_neurons LIF 
```

The environment '--env' can be "Ant-v4", "HalfCheetah-v4", "Walker2d-v4", "Hopper-v4", and "InvertedDoublePendulum-v4". The spiking neurons "--spiking_neurons" can be "LIF", "CLIF", "DN", and "ANN". To test the vanilla spiking actor network without the proxy network, set "--proxy" to False. Hyper-parameters can be modified with different arguments to main.py.
