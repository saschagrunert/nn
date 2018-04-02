-- | This module contains everything related to the main library interface
--
-- @since 0.1.0

module AI.Nn
  ( Network
  , predict
  , new
  , newIO
  , train
  ) where

import Data.List       (find
                       ,transpose)
import Data.List.Split (chunksOf)
import Data.Maybe      (fromJust)
import System.Random   (StdGen
                       ,getStdGen
                       ,randomRs)

-- | The network
--
-- @since 0.1.0
type Network = Network' ()

-- | The alias for a list of layers
--
-- @since 0.1.0
type Network' a = [Layer a]

-- | The network layer
--
-- @since 0.1.0
type Layer a = [(Neuron,a)]

-- | A network neuron
--
-- @since 0.1.0
data Neuron = Neuron { inputWeights :: [Double]      -- ^ The input weights
                     , activate :: Double -> Double  -- ^ The activation function
                     , activate' :: Double -> Double -- ^ The first derivation of the activation function
                     }

-- | The forward layer type
--
-- @since 0.1.0
data Forward = Forward { output :: Double
                       , sumInputWeight :: Double
                       , inputs :: [Double]
                       } deriving Show

-- | The alias for a list of input weights
--
-- @since 0.1.0
type Neuron' = [Double]

-- | The sigmoid activation function
--
-- @since 0.1.0
sigmoid :: Double -> Double
sigmoid x = 1.0 / (1 + exp (-x))

-- | The first derivation of the sigmoid function
--
-- @since 0.1.0
sigmoid' :: Double -> Double
sigmoid' x = sigmoid x * (1 - sigmoid x)

-- | Create a sigmoid neuron from given input weights
--
-- @since 0.1.0
sigmoidNeuron :: Neuron' -> Neuron
sigmoidNeuron ws = Neuron ws sigmoid sigmoid'

-- | Create a output neuron from given weights
--
-- @since 0.1.0
outputNeuron :: Neuron' -> Neuron
outputNeuron ws = Neuron ws id (const 1)

-- | Create a bias neuron from given number of inputs
--
-- @since 0.1.0
biasNeuron :: Int -> Neuron
biasNeuron i = Neuron (replicate i 1) (const 1) (const 0)

-- | Create a new Layer from a list of Neuron'
--
-- @since 0.1.0
createLayer :: Functor f => f t -> (t -> a) -> f (a, ())
createLayer n x = (\p -> (x p, ())) <$> n

-- | Create a new sigmoid Layer from a list of Neuron'
--
-- @since 0.1.0
sigmoidLayer :: [Neuron'] -> Layer ()
sigmoidLayer n = (biasNeuron x, ()) : createLayer n sigmoidNeuron
  where x = length $ head n

-- | Create a new output Layer from a list of Neuron'
--
-- @since 0.1.0
outputLayer :: [Neuron'] -> Layer ()
outputLayer n = createLayer n outputNeuron

-- | Create a new network for a StdGen and a number of layer and neurons
--
-- @since 0.1.0
new :: [Int] -> StdGen -> Network
new n g = (sigmoidLayer <$> init wss) ++ [outputLayer (last wss)]
 where
  rest                 = init n
  hiddenIcsNcs         = zip ((+ 1) <$> rest) (tail rest)
  (outputIc, outputNc) = (snd (last hiddenIcsNcs) + 1, last n)
  rs                   = randomRs (-1, 1) g
  (hidden, rs')        = foldl
    ( \(wss', rr') (ic, nc) ->
      let (sl, rs'') = pack ic nc rr' in (wss' ++ [sl], rs'')
    )
    ([], rs)
    hiddenIcsNcs
  (outputWss, _) = pack outputIc outputNc rs'
  wss            = hidden ++ [outputWss]
  pack ic nc ws = (take nc $ chunksOf ic ws, drop (ic * nc) ws)

-- | Create a new standard network for a number of layer and neurons
--
-- @since 0.1.0
newIO :: [Int] -> IO Network
newIO n = new n <$> getStdGen

-- | Do the complete back propagation
--
-- @since 0.1.0
backpropagate :: Network -> ([Double], [Double]) -> Network
backpropagate nw (xs, ys) = weightUpdate (forwardLayer nw xs) ys

-- | The learning rate
--
-- @since 0.1.0
rate :: Double
rate = 0.5

-- | Generate forward pass info
--
-- @since 0.1.0
forwardLayer :: Network -> [Double] -> Network' Forward
forwardLayer nw xs = reverse . fst $ foldl pf ([], 1 : xs) nw
 where
  pf (nw', xs') l = (l' : nw', xs'')
   where
    l'   = (\(n, _) -> (n, forwardNeuron n xs')) <$> l
    xs'' = (output . snd) <$> l'

-- | Generate forward pass info for a neuron
--
-- @since 0.1.0
forwardNeuron :: Neuron -> [Double] -> Forward
forwardNeuron n xs = Forward
  { output         = activate n net'
  , sumInputWeight = net'
  , inputs         = xs
  }
  where net' = calcNet xs (inputWeights n)

-- | Calculate the product sum
--
-- @since 0.1.0
calcNet :: [Double] -> [Double] -> Double
calcNet xs ws = sum $ zipWith (*) xs ws

-- | Updates the weights for an entire network
--
-- @since 0.1.0
weightUpdate
  :: Network' Forward
  -> [Double] -- ^ desired output value
  -> Network
weightUpdate fpnw ys = fst $ foldr updateLayer ([], ds) fpnw
  where ds = zipWith (-) ys ((output . snd) <$> last fpnw)

-- | Updates the weights for a layer
--
-- @since 0.1.0
updateLayer :: Layer Forward -> (Network, [Double]) -> (Network, [Double])
updateLayer fpl (nw, ds) = (l' : nw, ds')
 where
  (l, es) = unzip $ zipWith updateNeuron fpl ds
  ds' =
    map sum . transpose $ map (\(n, e) -> (* e) <$> inputWeights n) (zip l es)
  l' = (\n -> (n, ())) <$> l

-- | Updates the weights for a neuron
--
-- @since 0.1.0
updateNeuron :: (Neuron, Forward) -> Double -> (Neuron, Double)
updateNeuron (n, fpi) d = (n { inputWeights = ws' }, e)
 where
  e   = activate' n (sumInputWeight fpi) * d
  ws' = zipWith (\x w -> w + (rate * e * x)) (inputs fpi) (inputWeights n)


-- | Trains a network with a set of vector pairs until the global error is
-- smaller than epsilon
--
-- @since 0.1.0
train :: Double -> Network -> [([Double], [Double])] -> Network
train epsilon nw samples = fromJust
  $ find (\x -> globalQuadError x samples < epsilon) (trainUl nw samples)

-- | Create an indefinite sequence of networks
--
-- @since 0.1.0
trainUl :: Network -> [([Double], [Double])] -> [Network]
trainUl nw samples = iterate (\x -> foldl backpropagate x samples) nw

-- | Quadratic error for multiple pairs
--
-- @since 0.1.0
globalQuadError :: Network -> [([Double], [Double])] -> Double
globalQuadError nw samples = sum $ quadErrorNet nw <$> samples

-- | Quadratic error for a single vector pair
--
-- @since 0.1.0
quadErrorNet :: Network -> ([Double], [Double]) -> Double
quadErrorNet nw (xs, ys) =
  sum $ zipWith (\o y -> (y - o) ** 2) (predict nw xs) ys

-- | Calculates the output of a network for a given input vector
--
-- @since 0.1.0
predict :: Network -> [Double] -> [Double]
predict nw xs = foldl calculateLayer (1 : xs) nw
 where
  calculateLayer s = map (\(n, _) -> activate n (calcNet s (inputWeights n)))
