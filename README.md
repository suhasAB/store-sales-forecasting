# Project Title: Store Sales Prediction based on time series data
## Problem Statement:
Predicting the sale of 

## Professor: 
Carlos Rojas

## Term: 
Spring 2022

## Team Number: 
9

## Team Members:



<details>
     <summary> Click to see more details from previous deliverable</summary>
     What data you’ll use and where you’ll get it?  <br>
     - Scraping for all possible words on Sources tab of website (https://www.nytimes.com/games/wordle/index.html) <br>
     - Dataset involves upto 13,000 Valid English words with 5 letters each. <br>
     - Curated list of 2315 words published by Game Developer's partner.
     
</details> 

## 1.Abstract:
Wordle is an online word guessing game,originally created by Software Engineer Josh Wardle and currently published by The New York Times Company since 2022.wherein a new 5 letter word is set by the game each day and it is supposed to be guessed by players within 6 tries. User gets feedback about the closeness of his/her guess by 3 color indicators. Green on a block suggests the letter exists in the target word and is in the exact position the player has guessed, Yellow indicates the letter exists in the target word, but not in the position the player has guessed. Grey indicates the letter doesn't exist in the target word. Using these clues,Players are supposed to make better guesses in the remaining guesses to get to the target word in the minimum number of guesses.
We plan on analysing the game by exploring the trends and patterns in the 5 letter dataset,using Information theory concepts such as Entropy to deduce information gained from each guess and the best possible strategies to guess given the entropy calculated from the previous guesses. The goal is to come up with guesses having maximum entropy, i.e providing the most information so that we can use that information to make better guesses.This reduces the number of possible words in each turn. We also plan on exploring strategy of combining Entropy with Probability of Relative word Frequency of the possible words for guesses at each stage. The model should be able to come up with minimal and Informative guesses before reaching the target word.

### The game can be played in both interactive mode and automated mode,but Only Automated Games are considered for Evaluating Output Metrics
