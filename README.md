# nano_think
AI modelling for quick, local tasks

## Architecture

* Encoder
* Decoder
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
1. Encode
2. Add a few stored vectors to the context
3. Encode & attend
4. Generate output from each module
   * Transformer
   * Diffuser
   * State-Space
5. Combine expert outputs via weighted sum
6. Store the vector
7. Decode and output

* Everything is 384D


## Training

* Pre-train the transformer on `./data`
* Pre-train the diffuser on `./data`
* Pre-train the state-space model on `./data`

* Then do the whole graph to train just the MLA
