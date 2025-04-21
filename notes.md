1. 3blue1brown - what is neural network
    - great video, enough to run with the concept
2. Medium post 
    - great read, funny, enjoyable, got hung up up shape of matrices (optimized for math proofs in back prop vs conventional shape)
    - The concept of backprop from 10,000 ft clicked, but the math did not. Saved the backprop videos from 3blue1brown for later
3. Created blog with docusaurus
    - not the point of this, but great resource for spinning up simple JAMstack site
    - will look into hosting with github pages
4. Started Intro to deep learning with pytorch playlist
    - 1st video already taught me about github actions, linking colab notebooks to github
    - 2nd video, great intro to tensors, little slow
    - 3rd video, interesting aside on shape vs view, learned about strides
    - 4th video: was doing basic math and accidentally did XOR instead of exponent `torch.tensor([1,2,3,4]) ^ torch.tensor([5,6,7,8])` then was delighted to learn all about the AI winter and how even shallow networks with non-linearity are way more powerful than single-layer ones (Minsky Papert)
        ```python
        import torch
        import torch.nn as nn

        class XORNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(2, 2),
                    nn.ReLU(),
                    nn.Linear(2, 1),
                    nn.Sigmoid()
                )

            def forward(self, x):
                return self.net(x)
        ```
        - XOR is simplest non-linear function
 

