# nn
## A tiny neural network 🧠

This small neural network is based on the
[backpropagation](https://en.wikipedia.org/wiki/Backpropagation) algorithm.

## Usage

A minimal usage example would look like this:

```haskell
main :: IO ()
main = do
  {- Creates a new network with two inputs, two hidden layers and one output -}
  network <- newIO [2, 2, 1]

  {- Train the network for a common logical AND,
     until the maximum error of 0.01 is reached -}
  let trainedNetwork = train 0.01 network [([0, 0], [0])
                                          ,([0, 1], [0])
                                          ,([1, 0], [0])
                                          ,([1, 1], [1])]

  {- Predict the learned values -}
  let r00 = predict trainedNetwork [0, 0]
  let r01 = predict trainedNetwork [0, 1]
  let r10 = predict trainedNetwork [1, 0]
  let r11 = predict trainedNetwork [1, 1]

  {- Print the results -}
  putStrLn $ printf "0 0 -> %.2f" (head r00)
  putStrLn $ printf "0 1 -> %.2f" (head r01)
  putStrLn $ printf "1 0 -> %.2f" (head r10)
  putStrLn $ printf "1 1 -> %.2f" (head r11)
```

The result should be something like:

```console
0 0 -> -0.02
0 1 -> -0.02
1 0 -> -0.01
1 1 -> 1.00
```

## Hacking
To start hacking simply clone this repository and make sure that
[stack](https://docs.haskellstack.org/en/stable/README/) is installed. Then
simply hack around and build the project with:

```console
> stack build --file-watch
```

## Contributing
You want to contribute to this project? Wow, thanks! So please just fork it and
send me a pull request.
