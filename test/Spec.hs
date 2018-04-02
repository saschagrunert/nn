-- | The main test module
--
-- @since 0.1.0

module Main
  ( main
  ) where

import AI.Nn                 (new
                             ,predict
                             ,train)
import Test.Tasty            (TestTree
                             ,defaultMain
                             ,localOption
                             ,testGroup)
import Test.Tasty.Hspec      (Spec
                             ,it
                             ,parallel
                             ,shouldBe
                             ,testSpec)
import Test.Tasty.QuickCheck (QuickCheckTests (QuickCheckTests))

-- The main test routine
main :: IO ()
main = do
  uTests <- unitTests
  defaultMain . opts $ testGroup "Tests" [uTests]
  where opts = localOption $ QuickCheckTests 5000

-- Unit tests based on hspec
unitTests :: IO TestTree
unitTests = do
  actionUnitTests <- testSpec "Nn" nnSpec
  return $ testGroup "Unit Tests" [actionUnitTests]

-- Nn.hs related tests
nnSpec :: Spec
nnSpec = parallel $ do
  it "should succeed to train logical AND" $ do
    n <- new [2, 2, 1]
    let
      nw = train 0.001
                 n
                 [([0, 0], [0]), ([0, 1], [0]), ([1, 0], [0]), ([1, 1], [1])]
    round (head $ predict nw [1, 1]) `shouldBe` (1 :: Int)
    round (head $ predict nw [1, 0]) `shouldBe` (0 :: Int)
    round (head $ predict nw [0, 1]) `shouldBe` (0 :: Int)
    round (head $ predict nw [0, 0]) `shouldBe` (0 :: Int)

  it "should succeed to train logical OR" $ do
    n <- new [2, 2, 1]
    let
      nw = train 0.001
                 n
                 [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [1])]
    round (head $ predict nw [1, 1]) `shouldBe` (1 :: Int)
    round (head $ predict nw [1, 0]) `shouldBe` (1 :: Int)
    round (head $ predict nw [0, 1]) `shouldBe` (1 :: Int)
    round (head $ predict nw [0, 0]) `shouldBe` (0 :: Int)

  it "should succeed to train addition" $ do
    n <- new [2, 2, 1]
    let
      nw = train 0.001
                 n
                 [([0, 1], [1]), ([1, 1], [2]), ([1, 0], [1]), ([1, 2], [3])]
    round (head $ predict nw [0, 1]) `shouldBe` (1 :: Int)
    round (head $ predict nw [1, 0]) `shouldBe` (1 :: Int)
    round (head $ predict nw [1, 1]) `shouldBe` (2 :: Int)
    round (head $ predict nw [1, 2]) `shouldBe` (3 :: Int)
