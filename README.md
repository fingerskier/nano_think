# nano_think
AI modelling for quick, local tasks

## Architecture

* Multi-Head Latent attention
* Expert models
  * Transformer
    * 384D encoder-decoder
  * Diffuser
    * 384D
  * State-Space
    * 384D
* Vector store
  * 384D
* Pytorch implementation (GPU)


## Functionality

0. Input
1. Add a few stored vectors to the context
2. Encode & attend
3. Generate output from each module
   * Transformer
   * Diffuser
   * State-Space
4. Combine expert outputs via weighted sum
5. Store the vector
6. Decode and output

* Use the Transformer for encode/decode tasks
* Everything is 384D


## Training

* Pre-train the transformer on `./data`
* Pre-train the diffuser on `./data`
* Pre-train the state-space model on `./data`

* Then do the whole graph to train just the MLA


## Sleep

* Pruning
  * vectors are randomly sampled
  * search for highly similar vectors
  * create a new vectors by fusing the highly similar vectors
  * discard the original vectors
* Dreaming
  * vectors are randomly sampled
  * search for dissimilar vectors
  * create a new vector from the fusion of those vectors
  * retain the sampled vectors
