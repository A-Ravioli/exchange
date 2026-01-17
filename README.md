# exchange

i made clones of the NYSE and CME, originally for a research project with the very fun people at UMass Amherst's Quantum Information Theory lab, and am now open-sourcing it.

i built the original version in julia kinda quickly over a weekend, but julia kinda sucks as a language so we're rebuilding it in python so that more people can use it and so that i can wrap it in a gymnasium environment for some cool rl/ml natural HFT/MFT strategy discovery experiments which i show off in the experiments section!

i wrote a short blog post on how the exchanges work and it walks you through how to derive each feature logically (first principles-y and stuff). that blogpost will be linked here somewhere somehow eventually i hope. 

i will later finish rebuilding this in rust so that i can speed up the execution speed of this significantly to scale RL faster.

# how to make your way around the repo

here's a chill lil tree for the repo.

```bash
src/ // core python implementation
    file.py // it does a thing
    file.py // it does a thing
    file.py // it does a thing

src.jl/ // in julia
    file.jl // it does a thing
    file.jl // it does a thing
    file.jl // it does a thing

src.rs/ // in rust, so faster, i may delete this or have the python version wrap it
    file.rs // it does a thing, but faster
    file.rs // it does a thing, but faster
    file.rs // it does a thing, but faster
```

# making this a gym environment

i wanted to see if we could get 

# license

it's MIT licensed. don't do anything weird. the license is in [LICENSE](LICENSE)